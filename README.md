# Parallel Image Compression
## Abstract
A parallel implementation of the <b>DCT block-sorting lossless compression program</b> is described. The performance of the parallel implementation is compared to the sequential DCT program running on various shared-memory parallel architectures. The parallel DCT algorithm works by using the <i>forward discrete transform</i> followed by <i>quantization</i> and then <i>entropy encoding</i>. The output of the algorithm is fully compatible with the sequential version of DCT which is in wide use today.
## Results
The results show that a significant, near-linear speedup is achieved by using the parallel DCT program on systems with multiple processors. This will greatly reduce the time it takes to compress large amounts of data while remaining fully compatible with the sequential version of DCT.
<hr>
<i>Submitted in partial fulfillment of the requirements for CSE4001 : Parallel and Distributed Computing</i>
