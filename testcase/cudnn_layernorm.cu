#include "cudnn_common.cuh"
testcase::Result testcase::cudnn_layernorm(const Config& c) {
#if ORION_ENABLE_CUDNN
    dn::validate(c,4096,8);
    int64_t width=c.size, rows=8;
    auto x=dn::values(rows*width,c.seed);
    std::vector<float> scale(width), bias(width);
    for(int64_t j=0;j<width;++j) { scale[j]=1.0f+(j%5)/16.0f; bias[j]=(int(j%7)-3)/32.0f; }
    auto expected=dn_ref::layernorm(dn::doubles(x),scale,bias,width);
    Buffer<__half> dx(x.size()), out(expected.size());
    Buffer<float> ds(width), db(width);
    dn::upload(dx,x); dn::upload(ds,scale); dn::upload(db,bias);
    dn::Context ctx;
    auto X=ctx.tensor("X",{rows,width,1,1},{width,1,width,width});
    auto S=ctx.tensor("scale",{1,width,1,1},{width,1,width,width},dn::fe::DataType_t::FLOAT);
    auto B=ctx.tensor("bias",{1,width,1,1},{width,1,width,width},dn::fe::DataType_t::FLOAT);
    auto epsilon=ctx.graph.tensor(1e-5f);
    auto options=dn::fe::graph::Layernorm_attributes()
                 .set_forward_phase(dn::fe::NormFwdPhase_t::INFERENCE).set_epsilon(epsilon);
    auto outputs=ctx.graph.layernorm(X,S,B,options);
    auto Y=std::get<0>(outputs);
    Y->set_output(true).set_dim({rows,width,1,1}).set_stride({width,1,width,width});
    auto result=dn::execute(ctx,{{X,dx.ptr},{S,ds.ptr},{B,db.ptr},{Y,out.ptr}},out,expected,c.iterations,"cuDNN layernorm");
    dx.release(); ds.release(); db.release(); out.release(); return result;
#else
    (void)c; return dn::disabled();
#endif
}
