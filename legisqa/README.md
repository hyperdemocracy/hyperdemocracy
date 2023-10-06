# LegisQA

## Vector Store

streamlit will connect to a url that can resolve to the embedding stored in s3

streamlit -> hyperdemocracy.us/data -> cloudfront (maybe) + lambda -> s3://hyperdemocracy-<env>/data

```python
# db = lancedb.connect("s3://hyperdemocracy-<env>/data")
httpx.get("hyperdemocracy.us/data?query='foobar'")
```
