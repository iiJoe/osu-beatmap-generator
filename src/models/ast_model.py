from transformers import ASTModel, ASTConfig

config = ASTConfig(max_length=1024, patch_size=8, frequency_stride=8, time_stride=16)
ast_mdl = ASTModel(config)
