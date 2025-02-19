FROM ubuntu
RUN apt update
RUN apt install -y build-essential cmake nano git
ENV KOKKOS_VERSION=4.5.01
ENV KOKKOS_DOWNLOAD_URL=https://github.com/kokkos/kokkos/releases/download/${KOKKOS_VERSION}
WORKDIR /root
ADD ${KOKKOS_DOWNLOAD_URL}/kokkos-${KOKKOS_VERSION}.tar.gz ./
RUN tar -xvzf kokkos-${KOKKOS_VERSION}.tar.gz
RUN rm kokkos-${KOKKOS_VERSION}.tar.gz
WORKDIR kokkos-${KOKKOS_VERSION}
RUN cmake -B build \
-DCMAKE_CXX_COMPILER=g++ \
-DCMAKE_BUILD_TYPE=Release \
-DKokkos_ENABLE_OPENMP=ON \
-DKokkos_ARCH_NATIVE=ON \
-DKokkos_ENABLE_DEPRECATED_CODE_4=OFF
RUN cmake --build build
# install to /user/local
RUN cmake --install build 
WORKDIR /root
RUN rm -rf kokkos-${KOKKOS_VERSION}
# set these env vars to optimize OPENMP 4.0 performance
ENV OMP_PROC_BIND=spread
ENV OMP_PLACES=threads