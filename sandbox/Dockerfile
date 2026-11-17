FROM python:3.12-slim
RUN pip install --no-cache-dir pandas numpy matplotlib
ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl
WORKDIR /work