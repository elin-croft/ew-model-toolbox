from common.feature import BlockConfig

config=(
    BlockConfig(block_id="1", block_name="user",size=10, emb_size=8, common=True, dense=True),
    BlockConfig(block_id="2", block_name="item",size=10, emb_size=32, dense=True),
    BlockConfig(block_id="3", block_name="cross",size=10, emb_size=16, dense=True),
)