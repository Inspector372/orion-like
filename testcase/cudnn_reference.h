#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace testcase { namespace dn_ref {
inline std::vector<double> matmul(const std::vector<double>& a,
                                  const std::vector<double>& b, int m) {
    std::vector<double> y(m*m, 0.0);
    for (int i=0;i<m;++i) for (int j=0;j<m;++j)
        for (int k=0;k<m;++k) y[i*m+j] += a[i*m+k]*b[k*m+j];
    return y;
}
// NHWC input, KRSC filter, NHWK output; 3x3 cross-correlation, pad=1.
inline std::vector<double> convolution(const std::vector<double>& x,
                                       const std::vector<double>& w, int side) {
    constexpr int channels=8, outputs=8;
    std::vector<double> y(side*side*outputs, 0.0);
    for (int h=0;h<side;++h) for (int col=0;col<side;++col)
        for (int k=0;k<outputs;++k) for (int r=0;r<3;++r) for (int s=0;s<3;++s) {
            int ih=h+r-1, iw=col+s-1;
            if (ih<0 || iw<0 || ih>=side || iw>=side) continue;
            for (int c=0;c<channels;++c)
                y[(h*side+col)*outputs+k] += x[(ih*side+iw)*channels+c]*w[((k*3+r)*3+s)*channels+c];
        }
    return y;
}
inline std::vector<double> layernorm(const std::vector<double>& x,
                                     const std::vector<float>& scale,
                                     const std::vector<float>& bias, int width) {
    std::vector<double> y(x.size());
    for (std::size_t row=0;row<x.size()/width;++row) {
        double mean=0, var=0;
        for (int j=0;j<width;++j) mean+=x[row*width+j];
        mean/=width;
        for (int j=0;j<width;++j) { double d=x[row*width+j]-mean; var+=d*d; }
        var/=width;
        for (int j=0;j<width;++j)
            y[row*width+j]=(x[row*width+j]-mean)/std::sqrt(var+1e-5)*scale[j]+bias[j];
    }
    return y;
}
// B=1, H=2, D=64; noncausal attention without dropout or bias.
inline std::vector<double> attention(const std::vector<double>& q,
                                     const std::vector<double>& k,
                                     const std::vector<double>& v, int seq) {
    constexpr int heads=2, dim=64;
    std::vector<double> y(heads*seq*dim, 0.0), scores(seq);
    for (int h=0;h<heads;++h) for (int i=0;i<seq;++i) {
        for (int j=0;j<seq;++j) {
            double dot=0;
            for (int d=0;d<dim;++d) dot+=q[(h*seq+i)*dim+d]*k[(h*seq+j)*dim+d];
            scores[j]=dot/8.0;
        }
        double maximum=*std::max_element(scores.begin(),scores.end()), total=0;
        for (double& s:scores) { s=std::exp(s-maximum); total+=s; }
        for (int d=0;d<dim;++d) for (int j=0;j<seq;++j)
            y[(h*seq+i)*dim+d]+=scores[j]/total*v[(h*seq+j)*dim+d];
    }
    return y;
}
} }
