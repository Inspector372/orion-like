#include "cudnn_common.cuh"
testcase::Result testcase::cudnn_matmul(const Config& c) {
#if ORION_ENABLE_CUDNN
    dn::validate(c,512,8);
    int64_t m=c.size;
    auto a=dn::values(m*m,c.seed), b=dn::values(m*m,c.seed+1);
    auto expected=dn_ref::matmul(dn::doubles(a),dn::doubles(b),m);
    Buffer<__half> da(a.size()), db(b.size()), out(expected.size());
    dn::upload(da,a); dn::upload(db,b);
    dn::Context ctx;
    auto A=ctx.tensor("A",{1,m,m},{m*m,m,1});
    auto B=ctx.tensor("B",{1,m,m},{m*m,m,1});
    auto Y=ctx.graph.matmul(A,B,dn::fe::graph::Matmul_attributes());
    Y->set_output(true).set_dim({1,m,m}).set_stride({m*m,m,1});
    auto result=dn::execute(ctx,{{A,da.ptr},{B,db.ptr},{Y,out.ptr}},out,expected,c.iterations,"cuDNN matmul");
    da.release(); db.release(); out.release(); return result;
#else
    (void)c; return dn::disabled();
#endif
}
