FROM public.ecr.aws/lambda/python:latest
RUN dnf -y install git wget tar xz tree
RUN pip install --no-cache-dir onnxruntime
RUN pip install --no-cache-dir faster_whisper
RUN pip install --no-cache-dir ffmpeg-python
RUN wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz && tar xvf ffmpeg-release-arm64-static.tar.xz && mv ffmpeg-6.1-arm64-static/ff* /usr/bin && rm -Rf ffmpeg*
COPY model.py /tmp
RUN python /tmp/model.py && rm /tmp/model.py
RUN curl -O https://lambda-insights-extension-arm64.s3-ap-northeast-1.amazonaws.com/amazon_linux/lambda-insights-extension-arm64.rpm && \
    rpm -U lambda-insights-extension-arm64.rpm && \
    rm -f lambda-insights-extension-arm64.rpm
COPY split.py ${LAMBDA_TASK_ROOT}
COPY transcribe.py ${LAMBDA_TASK_ROOT}
COPY app.py ${LAMBDA_TASK_ROOT}
# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]
