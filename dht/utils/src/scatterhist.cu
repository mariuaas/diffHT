extern "C" __global__ 
void scatter_joint_hist(
    const long long* seg,
    const float* feats,
    const float* mesh_y,
    const float* mesh_x,
    const long long* featcombs,
    float* output,
    float* sigmaptr,
    const long long n,
    const long long nbins,
    const long long nfeats,
    const long long feat_dim
) {
    long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    long long nbins2 = nbins * nbins;
    float sigma = sigmaptr[0];
        
    if (idx < n) {
        long long s_idx = seg[idx];
        float y;
        float x;
        float z1;
        float z2;
        float value;
        long long j_y;
        long long j_x;
        
        for (long long j = 0; j < nfeats; j++){
            j_y = featcombs[2*j];
            j_x = featcombs[2*j+1];
            y = feats[idx*feat_dim + j_y];
            x = feats[idx*feat_dim + j_x];
            
            for (long long i = 0; i < nbins2; i++) {
                z1 = (y - mesh_y[i]) / sigma;
                z2 = (x - mesh_x[i]) / sigma;
                value = exp(-0.5 * (z1 * z1 + z2 * z2));
                atomicAdd(&output[s_idx * nfeats * nbins2 + j * nbins2 + i], value);
            }
        }
    }
}