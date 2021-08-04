---
layout: post
title:  "Wenet-cache"
date:   2021-08-04 10:00:00 +0800
categories: asr
---

# 进阶内容:cache


标准的forward是整个序列进行计算，但是在真正流式推断时，需要chunk级别的forward，因此需要引入cache的概念，即当前chunk的进行前向计算时，需要拿到上次前向的一些结果作为输入。


# Chunk-based方法
Cache涉及的


# Cache
**所有chunk size均指经过降采样后的解码帧，而不是输入帧。**

什么是cache？

对于流式推断，输入是一个个chunk的到来，对第i个chunk，当计算第k层网络的输出时，由于网络结构存在对左侧上下文的依赖，需要依赖第k-1层网络里在i之前的一些chunks的输出。
如果对于当前到来chunk，将其和之前的整个chunk序列拼起来作为网络输入进行前向，理论上没问题，但是其计算量会随着整个序列的长度和线性增长。
对于那些已经计算过的chunk的，可以将那些在计算下一个chunk的输出时需要的中间量保存下来，从而避免重新计算。这种方式就叫cache。

另外，wenet的网络在设计时，对于因果卷积和self-attention的左侧上下文都使用有限长度，因此无论序列多长，每次cache的大小是不变的（不增长）。


wenet/transformer/asr_model.py
用于，其内部也是使用了encoder.py中的forward_chunk(）函数。
```
@torch.jit.export
    def forward_encoder_chunk()
```

wenet/transformer/encoder.py

forward_chunk_by_chunk(）是python推断中使用按chunk依次计算的接口，该方法的结果，和送入整个序列通过mask进行流式模拟的结果应该是一致的。
其内部调用的forward_chunk(）函数。

forward_chunk(）是送入单个chunk进行前向的方法。
```
def forward_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        subsampling_cache: Optional[torch.Tensor] = None,
        elayers_output_cache: Optional[List[torch.Tensor]] = None,
        conformer_cnn_cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor],
               List[torch.Tensor]]:
```

由于单个chunk在计算时需要之前的chunk的计算得到的信息，因此这里保存了几钟需要的cache信息。
* required_cache_size是self-attention是对左侧的依赖长度，即subsampling_cache和elayers_output_cache的cache大小。
* conformer_cnn_cache的大小和required_cache_size无关，和casual网络的左侧上下文相关。


对于self-attention的所有缺失的输入都要cache。
对于

required_cache_size = decoding_chunk_size * num_decoding_left_chunks
decoding_chunk_size=4
num_decoding_left_chunks=2

num_left_chunks 
int requried_cache_size = opts_.chunk_size * opts_.num_left_chunks;

### offset
当按chunk进行输入时，不能直接得到chunk在序列中的位置，需要传入offset给出该chunk在整个序列里的偏移，从而可以计算positional encoding的位置。
xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)


### subsampling内部
subsampling内部的计算不进行cache。



### subsampling_cache
对subsampling的输出进行cache。
也就是第一个conformer block的输入。
```
if subsampling_cache is not None:
    cache_size = subsampling_cache.size(1)
    xs = torch.cat((subsampling_cache, xs), dim=1)
else:
    cache_size = 0
pos_emb = self.embed.position_encoding(offset - cache_size, xs.size(1))
if required_cache_size < 0:
    next_cache_start = 0
elif required_cache_size == 0:
    next_cache_start = xs.size(1)
else:
    next_cache_start = max(xs.size(1) - required_cache_size, 0)
r_subsampling_cache = xs[:, next_cache_start:, :]
```
图

### elayers_output_cache

对第1个到最后1个conformer block的输出进行cache。

也就是第2个conformer block的输入和conformer block之后的一层的输入。
```
for i, layer in enumerate(self.encoders):
    xs, _, new_cnn_cache = layer(xs,
        masks,
        pos_emb,
        output_cache=attn_cache,
        cnn_cache=cnn_cache)
    r_elayers_output_cache.append(xs[:, next_cache_start:, :])
```
output_cache并不参与计算，用来获取cache长度，
```
if output_cache is None:
    x_q = x
else:
    chunk = x.size(1) - output_cache.size(1)
    x_q = x[:, -chunk:, :]
    residual = residual[:, -chunk:, :]
    mask = mask[:, -chunk:, :]
```

mask和left_number_chunk有关。选择最后chunk大小的xs去和x做attention。
```
if output_cache is not None:
            x = torch.cat([output_cache, x], dim=1)
```
图


### conformer_cnn_cache
每个conformer block里的conv层的输入。

```
x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
```
```
if self.lorder > 0:
    if cache is None:
        x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
    else:
        x = torch.cat((cache, x), dim=2)
    assert (x.size(2) > self.lorder)
    new_cache = x[:, :, -self.lorder:]
```
cache大小为因果卷积左侧依赖的大小lorder。


图
