FROM python:3.12-slim

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
ENV PYTHONPATH=/home/user/app/src

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

CMD ["python", "-m", "fashion_how_graphdb.search_server", "--host", "0.0.0.0", "--port", "7860"]
