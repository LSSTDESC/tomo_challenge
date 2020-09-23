FROM ubuntu:18.04
MAINTAINER joezuntz@googlemail.com
#Joe's note to himself.  Compile this with: docker build -t joezuntz/cosmosis-base .
#then docker push joezuntz/cosmosis-base

# Basic compilers and tools dependencies
RUN apt-get update -y && apt-get install -y gcc g++ gfortran \
    cmake swig wget make libopenblas-dev libchealpix-dev \
    pkg-config curl python3 python3-distutils python3-dev python3-pip \
    libcfitsio-dev  libfftw3-dev   git  libgsl-dev autoconf \
    && apt-get clean all

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10

# Manual installation of mpich seems to be required to work on NERSC
RUN mkdir /opt/mpich && cd /opt/mpich \
    && wget http://www.mpich.org/static/downloads/3.3.1/mpich-3.3.1.tar.gz \
    && tar xvzf mpich-3.3.1.tar.gz &&  cd mpich-3.3.1 && ./configure --disable-wrapper-rpath && make -j4 \
    && make install && rm -rf /opt/mpich



# Also need a manual install of mpi4py so that it uses the right libraries - pip seems not to work
RUN mkdir /opt/mpi4py && cd /opt/mpi4py \
    && wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.2.tar.gz \
    && tar -zxvf mpi4py-3.0.2.tar.gz && cd mpi4py-3.0.2 && python setup.py install \
    && rm -rf /opt/mpi4py


# Need to install manually because we want MPI-enabled version
RUN mkdir /opt/hdf && cd /opt/hdf \
    && wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz \
    && tar xvf hdf5-1.10.5.tar.gz \
    && cd hdf5-1.10.5 \
    && ./configure --enable-parallel --prefix=/usr/local CC=mpicc \
    && make \
    && make install \
    && rm -rf /opt/hdf



# Some env vars
ENV HDF5_MPI ON
ENV HDF5_USE_FILE_LOCKING FALSE
ENV LD_LIBRARY_PATH /usr/local/lib
ENV C_INCLUDE_PATH /usr/local/include


ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y && apt-get install -yq r-base-dev && apt-get clean all


RUN pip install --no-cache-dir --upgrade pip

# Basic python requirements
RUN pip install --no-cache-dir numpy==1.19.2
RUN pip install --no-cache-dir cffi==1.14.2
RUN pip install --no-cache-dir Cython==0.29.21
RUN pip install --no-cache-dir pyyaml==5.3.1
RUN pip install --no-cache-dir pandas==1.1.2
RUN pip install --no-cache-dir setuptools==50.3.0
RUN pip install --no-cache-dir click==7.1.2
RUN pip install --no-cache-dir progressbar==2.5
RUN pip install --no-cache-dir --no-binary=h5py h5py==2.10.0
RUN pip install --no-cache-dir matplotlib==3.3.2
RUN pip install --no-cache-dir fitsio==1.1.2
RUN pip install --no-cache-dir scipy==1.5.2
RUN pip install --no-cache-dir scikit-learn==0.23.2
RUN pip install --no-cache-dir astropy==4.0.1.post1
RUN pip install --no-cache-dir rpy2==3.3.5
RUN pip install --no-cache-dir tensorflow==2.3.0
RUN pip install --no-cache-dir keras==2.4.3
RUN pip install --no-cache-dir jax==0.1.77
RUN pip install --no-cache-dir pyccl==2.1.0
RUN pip install --no-cache-dir sacc==0.4.5
RUN pip install --no-cache-dir camb==1.1.3
RUN pip install --no-cache-dir cosmosis-standalone==0.2.2
RUN pip install --no-cache-dir flax==0.2.1
RUN pip install --no-cache-dir --pre jax-cosmo==0.1rc6
RUN pip install --no-cache-dir git+https://github.com/LSSTDESC/firecrown/@v0.4
RUN apt-get update \
    && apt-get install -y software-properties-common \
    && apt-get clean all

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin \
    && mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub \
    && add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" \
    && apt-get update \
    && apt-get -y install cuda-runtime-10-1 \
    && apt-get clean all

# # stuff RPY2 needs
RUN apt-get install -y openssl libssl-dev libgit2-dev libcurl4 libxml2-dev
# RUN apt-get install -y openssl  \
# RUN apt-get install -y openssl libcurl4-gnutls-dev libgit2-dev \
#     && apt-get clean all

# # first-time import installs a bunch of stuff
COPY rpy_installer.py  /opt/rpy_installer.py
RUN cd /opt && python rpy_installer.py && rm /opt/rpy_installer.py

RUN pip install --no-cache-dir keras-tcn==3.1.1
RUN pip install --no-cache-dir git+https://github.com/dkirkby/zotbin@2d290e198b3baaf3bebd6292045e2f691748210d
RUN pip install --no-cache-dir lightgbm==2.3.1

RUN pip uninstall -y jax-cosmo
RUN pip install git+https://github.com/DifferentiableUniverseInitiative/jax_cosmo

# These prevent errors all over the place on NERSC
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
CMD ["/bin/bash", "-l"]


