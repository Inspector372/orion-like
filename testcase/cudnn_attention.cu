#include "cudnn_common.cuh"
testcase::Result testcase::cudnn_attention(const Config& c) {
#if ORION_ENABLE_CUDNN
    dn::validate(c,512,64);
    int64_t seq=c.size, heads=2, dim=64;
    auto q=dn::values(heads*seq*dim,c.seed), k=dn::values(heads*seq*dim,c.seed+1),
         v=dn::values(heads*seq*dim,c.seed+2);
    auto expected=dn_ref::attention(dn::doubles(q),dn::doubles(k),dn::doubles(v),seq);
    Buffer<__half> dq(q.size()), dk(k.size()), dv(v.size()), out(expected.size());
    dn::upload(dq,q); dn::upload(dk,k); dn::upload(dv,v);
    dn::Context ctx;
    std::vector<int64_t> shape={1,heads,seq,dim}, stride={heads*seq*dim,seq*dim,dim,1};
    auto Q=ctx.tensor("Q",shape,stride), K=ctx.tensor("K",shape,stride), V=ctx.tensor("V",shape,stride);
    auto options=dn::fe::graph::SDPA_attributes().set_is_inference(true)
                 .set_causal_mask(false).set_attn_scale(0.125f);
    auto outputs=ctx.graph.sdpa(Q,K,V,options);
    auto Y=std::get<0>(outputs);
    Y->set_output(true).set_dim(shape).set_stride(stride);
    auto result=dn::execute(ctx,{{Q,dq.ptr},{K,dk.ptr},{V,dv.ptr},{Y,out.ptr}},out,expected,c.iterations,"cuDNN SDPA");
    dq.release(); dk.release(); dv.release(); out.release(); return result;
#else
    (void)c; return dn::disabled();
#endif
}
