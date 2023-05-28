import pinecone

index_name = 'openai-youtube-transcriptions'

# initialize connection (get API key at app.pinecone.io)
pinecone.init(
    api_key="YOUR_API_KEY",
    environment="YOUR_ENV"  # find next to API key
)

# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='cosine',
        metadata_config={
            'indexed': ['channel_id', 'published']
        }
    )
# connect to index
index = pinecone.Index(index_name)
# view index stats
index.describe_index_stats()