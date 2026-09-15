#include "cudnn_common.cuh"
testcase::Result testcase::cudnn_convolution(const Config& c) {
#if ORION_ENABLE_CUDNN
    dn::validate(c,128);
    int64_t s=c.size, channels=8;
    auto x=dn::values(s*s*channels,c.seed), w=dn::values(8*3*3*8,c.seed+1);
    auto expected=dn_ref::convolution(dn::doubles(x),dn::doubles(w),s);
    Buffer<__half> dx(x.size()), dw(w.size()), out(expected.size());
    dn::upload(dx,x); dn::upload(dw,w);
    dn::Context ctx;
    auto X=ctx.tensor("X",{1,8,s,s},{s*s*8,1,s*8,8});
    auto W=ctx.tensor("W",{8,8,3,3},{72,1,24,8});
    auto options=dn::fe::graph::Conv_fprop_attributes().set_padding({1,1})
                 .set_stride({1,1}).set_dilation({1,1});
    auto Y=ctx.graph.conv_fprop(X,W,options);
    Y->set_output(true).set_dim({1,8,s,s}).set_stride({s*s*8,1,s*8,8});
    auto result=dn::execute(ctx,{{X,dx.ptr},{W,dw.ptr},{Y,out.ptr}},out,expected,c.iterations,"cuDNN convolution");
    dx.release(); dw.release(); out.release(); return result;
#else
    (void)c; return dn::disabled();
#endif
}
