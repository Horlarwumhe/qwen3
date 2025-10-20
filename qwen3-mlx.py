
import functools
import mlx.core as mx
from qtoken import Qwen3Tokenizer
import random
import time

TIMES = {}
QUANTIZED=True
BITS=4
FAST_MODE = True


def measure(fn):
    @functools.wraps(fn)
    def inner(*a, **b):
        start = time.time()
        r = fn(*a, **b)
        end = time.time() - start
        key = fn.__qualname__
        if key in TIMES:
            TIMES[key] += end
        else:
            TIMES[key] = end  
        return r
    return inner


def silu(x):
    if FAST_MODE:
        return x * mx.sigmoid(x) 
    return x * (1 / (1 + mx.exp(-x)))  # x * sigmoid(x)


def softmax(x, axis=-1):
    return mx.softmax(x,-1)
    # e_x = mx.exp(x - mx.max(x, axis=axis, keepdims=True))
    # return e_x / e_x.sum(axis=axis, keepdims=True)

class Module:

    def __call__(self, *args, **kwds):
        return self.forward(*args,**kwds)

class Linear(Module):

    def __init__(self, in_features, out_features, dtype=None, bias=True):
        super().__init__()
        bound = mx.sqrt(1 / in_features)
        self.weight = mx.random.uniform(-bound,bound, (out_features, in_features))
        self._shape = (out_features,in_features)
        if bias:
            self.bias = mx.random.uniform(-bound,bound, (out_features,)) 
        else:
            self.bias = None
        self.bits = BITS

    def forward(self,x):
        if isinstance(self.weight,tuple):
            return mx.quantized_matmul(x,*self.weight,bits=self.bits) + (self.bias if self.bias is not None else 0)
        return (x @ self.weight.T + (self.bias if self.bias is not None else 0))

    @property
    def shape(self):
        return self._shape
    
class MoeLinear(Linear):
    
    def __init__(self,in_features,out_features,num_experts,bias=False):
        bound = mx.sqrt(1 / in_features)
        self._shape = (num_experts,out_features,in_features)
        self.weight = mx.random.uniform(-bound,bound, (num_experts,out_features, in_features))
        if bias:
            self.bias = mx.random.uniform(-bound,bound, (out_features,)) 
        else:
            self.bias = None

    def forward(self,x):
        if x.ndim > 2:
            x = x[...,None,:,:] # (..., num_expert, y, z)
        x = (x @ self.weight.swapaxes(1,2) + (self.bias if self.bias is not None else 0))
        return x.swapaxes(0,1) if x.ndim > 2 else x # (num_experts, .....)
    
    @property
    def shape(self):
        return self.weight.shape

class Drop(Module):
    def __init__(self, r=0.9):
        # different from torch.nn.Dropout. This drop values that are less than p*max(x). drop means set to 0
        super().__init__()
        self.r = r

    def forward(self,x):
        return mx.where(x > mx.max(x,axis=-1,keepdims=True) * self.r, x, 0)


class Embedding(Module):

    def __init__(self,num_embeddings, dim, dtype=mx.float32):
    
        super().__init__()
        self.dtype = dtype
        self.weight = mx.random.normal((num_embeddings, dim),None,0,1) #astype(dtype)

    def forward(self, x):
       return self.weight[x].astype(self.dtype)
    

class FeedForward(Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.up_proj = Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.down_proj = Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)
    
    def forward(self, x):
        x_gate = self.gate_proj(x)
        x_up = self.up_proj(x)
        x = silu(x_gate) * x_up
        x = self.down_proj(x)
        return x


class MoeFeedForward(Module):

    def __init__(self, cfg):
        super().__init__()
        num_experts = cfg.get("num_expert")
        self.gate_proj = MoeLinear(cfg["emb_dim"], cfg["moe_size"],num_experts,bias=False)
        self.up_proj = MoeLinear(cfg["emb_dim"], cfg["moe_size"], num_experts,bias=False)
        self.down_proj = MoeLinear(cfg["moe_size"], cfg["emb_dim"], num_experts, bias=False)
    
    @measure
    def forward(self, x):
        x_gate = self.gate_proj(x)
        x_up = self.up_proj(x)
        x = silu(x_gate) * x_up
        x = self.down_proj(x)
        return x

class RMSNorm(Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.weight = mx.ones(emb_dim)
        self.shift = mx.zeros(emb_dim) if bias else 0
    @measure
    def forward(self, x):
        '''
        Y = (x/ sqrt(RMS(x) +e )) * y
        RMS = (sum(x^2)/n)
        '''
        input_dtype = x.dtype
        if self.qwen3_compatible:
            x = x.astype(mx.float32)
        if FAST_MODE:
            return (mx.fast.rms_norm(x,self.weight,self.eps) + self.shift ).astype(input_dtype)
        else:
            rms = mx.mean(mx.power(x,2),axis=-1,keepdims=True)
            x =  (x / (mx.sqrt(rms + self.eps))) * self.weight
            if self.shift:
                x = x + self.shift
            return x.astype(input_dtype)

class RoPE(Module):
    def __init__(self, head_dim, theta_base=10_000, context_length=4096, dtype=mx.float32):
        super().__init__()
        self.head_dim = head_dim
        self.theta_base = theta_base
        self.context_length = context_length
        self.cos, self.sin = self.compute_rope_params(
            head_dim=self.head_dim,
            theta_base=self.theta_base,
            context_length=self.context_length,
            dtype=dtype
        )
    @measure
    def forward(self, x,offset=0):
        
        # x: (batch_size, num_heads, seq_len, head_dim)
        if FAST_MODE:
            return mx.fast.rope(x,x.shape[-1],offset=offset,base=self.theta_base,traditional=False,scale=1)
        cos, sin = self.cos, self.sin
        # print(cos.shape,sin.shape,"cos-sin")
        batch_size, num_heads, seq_len, head_dim = x.shape
        assert head_dim % 2 == 0, "Head dimension must be even"

        # Split x into first half and second half
        x1 = x[..., : head_dim // 2]  # First half
        x2 = x[..., head_dim // 2 :]  # Second half
        rotated = mx.concatenate((-x2,x1), axis=-1)
        cos = cos[offset:seq_len+offset,...]
        sin = sin[offset:seq_len+offset,...]
        x_rotated = (x * cos) + (rotated * sin)
        # It's ok to use lower-precision after applying cos and sin rotation
        x = x_rotated.astype(x.dtype)
        return x
    
    
    def compute_rope_params(self,head_dim, theta_base=10_000, context_length=4096, dtype=mx.float32):
        assert head_dim % 2 == 0, "Embedding dimension must be even"
        # head_dim = head_dim*2
        # Compute the inverse frequencies
        inv_freq = 1.0 / (theta_base ** (mx.arange(0, head_dim,2, dtype=dtype)[: (head_dim // 2)].astype(mx.float32) / head_dim))
        inv_freq = inv_freq.reshape(1,head_dim//2)
        # Generate position indices
        positions = mx.arange(context_length, dtype=dtype).reshape(context_length,1)

        # Compute the angles
        angles = positions * inv_freq  # Shape: (context_length, head_dim // 2)

        # Expand angles to match the head_dim
        angles = mx.concatenate([angles, angles], axis=1)  # Shape: (context_length, head_dim)

        # Precompute sine and cosine
        cos = mx.cos(angles)
        sin = mx.sin(angles)

        return cos, sin



class GroupedQueryAttention(Module):
    def __init__(
        self, emb_dim, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None,rope_params=None,use_kv_cache=False
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert emb_dim % num_heads == 0, "`emb_dim` must be divisible by `num_heads` if `head_dim` is not set"
            head_dim = emb_dim // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.q_proj = Linear(emb_dim, self.d_out, bias=False, dtype=dtype)
        self.k_proj = Linear(emb_dim, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.v_proj = Linear(emb_dim, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = Linear(self.d_out, emb_dim, bias=False, dtype=dtype)
        self.rope = RoPE(head_dim=head_dim,theta_base=rope_params["theta_base"],
                         context_length=rope_params["context_length"], dtype=dtype)
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None
        self.kv_cache = None
        self.use_kv_cache = use_kv_cache
        self.softmax_scale = 1/mx.sqrt(self.head_dim)

    @measure
    def forward(self, x, mask):
        b, num_tokens, _ = x.shape
        # Apply projections
        # (b, num_tokens, num_heads * head_dim)
        queries = self.q_proj(x)
         # (b, num_tokens, num_kv_groups * head_dim)
        keys = self.k_proj(x)
         # (b, num_tokens, num_kv_groups * head_dim)
        values = self.v_proj(x)

        # Reshape
        queries = queries.reshape(b, num_tokens, self.num_heads, self.head_dim).swapaxes(1, 2)
        keys = keys.reshape(b, num_tokens, self.num_kv_groups, self.head_dim).swapaxes(1, 2)
        values = values.reshape(b, num_tokens, self.num_kv_groups, self.head_dim).swapaxes(1, 2)

        # Optional normalization
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)
        # Apply RoPE
        offset = 0
        if self.kv_cache:
            offset = self.kv_cache[0].shape[2]

        queries = self.rope(queries,offset)
        keys = self.rope(keys,offset)

        if self.use_kv_cache:
            if self.kv_cache:
                k,v = self.kv_cache
                # k,v = mx.dequantize(*k),mx.dequantize(*v)
                keys = mx.concatenate([k, keys], axis=2)
                values = mx.concatenate([v, values], axis=2)
                self.kv_cache = (keys,values)
                # self.kv_cache = (mx.quantize(keys),mx.quantize(values))
            else:
                self.kv_cache = (keys,values)
                # self.kv_cache = (mx.quantize(keys),mx.quantize(values))

        
        # Attention softmax(Q @ K.T, dim=-1) @ V
        if FAST_MODE:
            context = mx.fast.scaled_dot_product_attention(
                 queries, keys, values, mask=~mask, scale=self.softmax_scale)
            # context = context.astype()
        else:
        # Expand K and V to match number of heads
            keys = mx.repeat(keys,self.group_size, axis=1)
            values = mx.repeat(values,self.group_size, axis=1)
            attn_scores = queries @ keys.swapaxes(2, 3)
            attn_scores = mx.where(mask,-mx.inf,attn_scores)
            attn_weights = softmax(attn_scores * self.softmax_scale, axis=-1)
            context = (attn_weights @ values)
        out = self.out_proj(context.swapaxes(1, 2).reshape(b, num_tokens, -1))
        return out
    

class TransformerBlock(Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            emb_dim=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            head_dim=cfg["head_dim"],
            num_kv_groups=cfg["n_kv_groups"],
            qk_norm=cfg["qk_norm"],
            dtype=cfg["dtype"],
            use_kv_cache = cfg.get("use_kv_cache",False),
            rope_params={
                 "head_dim": cfg["head_dim"], 
                 "theta_base": cfg["rope_base"], 
                 "context_length": cfg["context_length"]}
        )
        self.ff = FeedForward(cfg)
        self.input_layernorm  = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.post_attention_norm = RMSNorm(cfg["emb_dim"], eps=1e-6)

    @measure
    def forward(self, x, mask):
        out = self.input_layernorm(x)
        # Add the original input back
        out = self.att(out,mask) + x
        x = out
        out = self.post_attention_norm(out)
        
        # Add the original input back
        out = self.ff(out) + x 
        return  out



class Qwen3Model(Module):
    def __init__(self, cfg):
        super().__init__()

        # Main model parameters
        self.tok_emb = Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        self.kv_cache = [None for _ in range(cfg["n_layers"])]

        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        self.cfg = cfg

    @measure
    def forward(self, in_idx):
        # print(in_idx)
        x = self.tok_emb(in_idx)
        b, num_tokens, _ = x.shape
        has_kv = self.trf_blocks[0].att.kv_cache and self.trf_blocks[0].att.use_kv_cache
        if has_kv:
            kv = self.trf_blocks[0].att.kv_cache
            past_len = kv[0].shape[2]
            seq_len = past_len + num_tokens
            mask =  mx.triu(mx.ones((num_tokens, seq_len),mx.bool_), k=past_len+1)
        else:
            padding_mask = (in_idx == tokenizer.pad_token_id).reshape(b, 1, 1, num_tokens)
            mask =  mx.triu(mx.ones((num_tokens, num_tokens),mx.bool_), k=1).reshape(1, 1, num_tokens, num_tokens)# (batch,num_tokens,num_tokens)
            mask =  mask | padding_mask
        for i,block in enumerate(self.trf_blocks):
            block.layer_id = i + 1
            block.att.layer_id = i+1
            x = block(x, mask)
        x = self.final_norm(x)
        logits = self.out_head(x.astype(self.cfg["dtype"]))
        return logits


def load_weights_into_qwen(model, param_config, params):
    dtype = param_config['dtype']

    def assign(left, right, tensor_name="unknown",quantize=True):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")
        if right.dtype != dtype:
            return mx.array(right).astype(dtype)
        bits = param_config.get('bits',None)
        if bits and right.ndim > 1 and quantize:
            return mx.quantize(right,bits=bits)
        return right
    
    def assing_quantize(left, right, tensor_name="unknown",quantize=True):
        # This works only for 4bit quantization. Will need to be modified for other bits.
        # expected_shape = ((right.shape[0], right.shape[1] * 8) if right.ndim > 1 else right.shape)
        # if left.shape != expected_shape:
        #     raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {expected_shape}")
        bits = param_config.get('bits',None)
        if right.ndim == 1:
            return right
        tensor_name = tensor_name.replace('.weight','')
        scale = params[tensor_name+".scales"]
        bias = params[tensor_name+".biases"]
        if not quantize:
            # dont quantize
            return mx.dequantize(right,scale,bias,bits=bits)
        return (right,scale,bias)

    if "model.embed_tokens.scales" in params:
        fn = assing_quantize
        print("weight qunatized 4bit.")
    else:
        fn = assign
    
    model.tok_emb.weight = fn(model.tok_emb.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight",False)
    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]
        attn = block.att

        # Q, K, V projections
        for proj in ('q_proj',"k_proj","v_proj","out_proj"):
            att_proj = getattr(attn,proj)
            key = "o_proj" if proj == "out_proj" else proj
            att_proj.weight = fn(
                att_proj.weight,
                params[f"model.layers.{l}.self_attn.{key}.weight"],
                f"model.layers.{l}.self_attn.{key}.weight"
            )
        # Q,K norm
        for norm in ("q_norm","k_norm"):
            att_norm = getattr(attn,norm)
            if att_norm is not None:
                att_norm.weight = fn(
                      att_norm.weight,
                      params[f"model.layers.{l}.self_attn.{norm}.weight"],
                      f"model.layers.{l}.self_attn.{norm}.weight"
                )
            
        block.input_layernorm.weight = assign(
            block.input_layernorm.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Feedforward(MLP) weights
        for ff_proj in ("gate_proj", "up_proj", "down_proj"):
            ff = getattr(block.ff,ff_proj)
            ff.weight = fn(
                    ff.weight,
                    params[f"model.layers.{l}.mlp.{ff_proj}.weight"],
                    f"model.layers.{l}.mlp.{ff_proj}.weight"
                )
        block.post_attention_norm.weight = fn(
            block.post_attention_norm.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Final normalization and output head
    model.final_norm.weight = fn(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params:
        model.out_head.weight = fn(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        print("Model uses weight tying.")
        model.out_head.weight = fn(model.out_head.weight, params["model.embed_tokens.weight"], "model.embed_tokens.weight")
    mx.eval()

def generate_text_basic_stream(model, token_ids, max_new_tokens,r=0.90, eos_token_id=None):
    # model.eval()
    print(f"{token_ids.shape=}")
    batch_size = token_ids.shape[0]
    sample_size = 5 # sample 5 token
    # Generate subsequent tokens using cache
    DROP=True
    drop = Drop(r=r)
    if model.trf_blocks[0].att.use_kv_cache:
        print("using kv cache")
    else:
        print("No kv cache")
    generation_stream = mx.new_stream(mx.default_device())
    with mx.stream(generation_stream):
        for n in range(max_new_tokens):
            out = model(token_ids)
            out = out[:, -1] # Pick last token.
            import pdb
            # pdb.set_trace()
            if DROP:
                # apply noise to shuffle scores
                out = mx.abs(drop(out) * mx.random.normal((out.shape[-1],)))
                mx.eval(out)
                next_token = mx.argmax(out,-1,keepdims=True)
            else:
                # use simple random sampling
                mx.eval(out)
                i = random.randint(0,sample_size-1)
                next_token = mx.argsort(out, axis=-1)[...,-sample_size:] # shape (batch_size, sample_size) 5 tokens to pick from
                next_token =  next_token[...,i].reshape(-1,1)
            if (eos_token_id is not None
                    and mx.all(next_token == eos_token_id)):
                break

            yield next_token
            if model.trf_blocks[0].att.use_kv_cache:
                token_ids = next_token
            else:
                token_ids = mx.concatenate([token_ids, next_token], axis=1)
            mx.clear_cache()

        
def create_input(*prompts):
    prompts = [tokenizer.encode(prompt) for prompt in prompts]
    max_len = len(max(prompts,key=lambda x: len(x)))
    for prompt in prompts:
        padding = max_len - len(prompt)
        if padding > 0:
            prompt.extend([tokenizer.pad_token_id]*padding)
    token_ids = mx.array(prompts)
    return token_ids

def get_config(model):
    if model == "0.6B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,           # Vocabulary size
            "context_length": 40_960,        # Context length that was used to train the model
            "emb_dim": 1024,                 # Embedding dimension
            "n_heads": 16,                   # Number of attention heads
            "n_layers": 28,                  # Number of layers
            "hidden_dim": 3072,              # Size of the intermediate dimension in FeedForward
            "head_dim": 128,                 # Size of the heads in GQA
            "qk_norm": True,                 # Whether to normalize queries and values in GQA
            "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
            "rope_base": 1_000_000.0,        # The base in RoPE's "theta"
            "dtype": mx.float16,         # Lower-precision dtype to reduce memory usage
            "use_kv_cache":True,
            "weight": "Qwen3-0.6B-Base/model.safetensors",
            "tokenizer":"Qwen3-0.6B-Base/tokenizer.json",
        }

    elif model == "1.7B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 2048,                 # 2x larger than above
            "n_heads": 16,
            "n_layers": 28,
            "hidden_dim": 6144,              # 2x larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": mx.float16,
            "use_kv_cache":True,
            "weight": "Qwen3-1.7B-Base/model.safetensors",
            "tokenizer":"Qwen3-1.7B-Base/tokenizer.json"
        }   

    elif model == "4B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 2560,                 # 25% larger than above
            "n_heads": 32,                   # 2x larger than above
            "n_layers": 36,                  # 29% larger than above
            "hidden_dim": 9728,              # ~3x larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": mx.float16,
            "use_kv_cache":True,
            "weight": "Qwen3-4B/model.safetensors",
            "tokenizer":"Qwen3-4B/tokenizer.json",
            "3bit":"Qwen3-4B/model3bit.safetensors"
        } 
    return QWEN3_CONFIG

def load_model(config):
    model = Qwen3Model(config)
    if BITS == 3 and "3bit" in config:
        file = config['3bit']
    else:
        file = config['weight']
    print("using model file ",file)
    weight = mx.load(file,format="safetensors")
    print("loading into model")
    load_weights_into_qwen(model, config, weight)
    del weight
    print("model loaded")
    return model

def load_tokenizer(config):
    USE_REASONING_MODEL = True
    tokenizer = Qwen3Tokenizer(
        tokenizer_file_path=config['tokenizer'],
        add_generation_prompt=USE_REASONING_MODEL,
        add_thinking=USE_REASONING_MODEL
    )
    return tokenizer


def create_argparse():
    import argparse

    parser = argparse.ArgumentParser(description="Model configuration")

    # Model selection with choices
    parser.add_argument("-m", "--model",type=str,choices=["0.6", "1.7","4"],required=True,help="Model"
    )
    parser.add_argument("-f", "--fast",action="store_true",help="Use fast mode (mlx.fast)"
    )
    parser.add_argument("-p", "--prompt",type=str,required=True,help="Input prompt"
    )
    parser.add_argument("-s", "--max-tokens",type=int,default=1000,help="Maximum number of tokens to generate"
    )
    parser.add_argument( "-r", "--drop-size",type=float,default=0.90,help="Drop size parameter"
    )
    parser.add_argument("-i ", "--interactive",action="store_true",help="Interactive mode")
    parser.add_argument("--no-kv",action="store_true",help="Disable KV cache")
    parser.add_argument("--bits",type=int,help="quatization bits [8,4]", default=4)
    parser.add_argument("-T","--nothink",action="store_true",help="Disable thinking mode")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = create_argparse()
    m = args.model+"B"
    config = get_config(m)
    bits = args.bits
    config['bits'] = bits
    BITS = bits
    config['use_kv_cache'] = not args.no_kv
    model = load_model(config)
    tokenizer = load_tokenizer(config)
    if not args.prompt:
        prompt = None
    else:
        if args.nothink:
            args.prompt = "/nothink " + args.prompt
        prompt = create_input(args.prompt)
    max_token = args.max_tokens
    drop_size = args.drop_size
    FAST_MODE = args.fast
    if FAST_MODE:
        print("Using FAST MODE")
    batch = prompt.shape[0] > 1 if prompt is not None else False
    count = total_time = 0
    import pdb
    tokens = []
    interrupt = False
    while True:
        emoji = []
        if prompt is not None:
            try:
                start = time.time()
                for token_batch in generate_text_basic_stream(
                            model=model,
                            token_ids=prompt,
                            max_new_tokens=max_token,
                            eos_token_id=tokenizer.eos_token_id,
                            r=drop_size
                            ):
                    
                    count += 1
                    if not batch:
                        token = token_batch.flatten().tolist()
                        if token[0] == tokenizer.think_start:
                            print("<think>\t\t")
                        elif token[0] == tokenizer.think_end:
                            print("\n</think>\n\n")
                        else:
                            s = tokenizer.decode(token)
                            if b"\xef\xbf\xbd" in s.encode():
                                emoji.append(token[0])
                                continue
                            if emoji:
                                print(tokenizer.decode(emoji),end="",flush=True)
                                emoji = []
                            print(s,end="",flush=True)
                            tokens.append(token[0])
                        
                    else:
                        tokens.append(token_batch)
                        print('.',end="",flush=True)
            except KeyboardInterrupt:
                interrupt = True
            else:
                interrupt = False
            total_time += time.time() - start
        if args.interactive:
            print("\n\nType next prompt (q to quit): ")
            text = input(">> ")
            if text.lower() == "q":
                break
            if not text and interrupt:
                prompt = mx.array(token).reshape(-1,1) 
                # continue generation after keyboard interrupt
            else:
                if args.nothink:
                    text = "/nothink " + text
                prompt = create_input(text)
            
            print("Generating response...\n")
        else:
            break
    kv = model.trf_blocks[0].att.kv_cache[0]
    kv_size = kv.nbytes
    kv_size = (kv_size * 2 * config['n_layers']) / (1024*1024)
    print("\ngeneration end ...... \ntokens=%s max=%s speed=%s"%(count,max_token,round(count/total_time)))
    print("kv(%sMB) size %s"%(kv_size,kv.shape))