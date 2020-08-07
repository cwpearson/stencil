#pragma once

namespace gpu_topo {

/* return a number representing the bandwidth between `src` and `dst`
*/
double bandwidth(int src, int dst);

/* Try to enable peer access between src and dst
   May accelerate calls to `peer`
*/
void enable_peer(const int src, const int dst);

/* src has peer access to dst
*/
bool peer(const int src, const int dst);

} // namespace gpu_topo
