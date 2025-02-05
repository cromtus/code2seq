class Config:
    @staticmethod
    def get_default_config(args):
        config = Config(args)
        config.NUM_EPOCHS = 3000
        config.SAVE_EVERY_EPOCHS = 1
        config.PATIENCE = 10
        config.BATCH_SIZE = 512
        config.TEST_BATCH_SIZE = 256
        config.READER_NUM_PARALLEL_BATCHES = 1
        config.SHUFFLE_BUFFER_SIZE = 10000
        config.CSV_BUFFER_SIZE = 100 * 1024 * 1024  # 100 MB
        config.MAX_CONTEXTS = 200
        config.SUBTOKENS_VOCAB_MAX_SIZE = 190000
        config.TARGET_VOCAB_MAX_SIZE = 27000
        config.EMBEDDINGS_SIZE = 256
        config.NUM_ATTENTION_HEADS = 1
        config.DECODER_SIZE = 320
        config.NUM_ENCODER_LAYERS = 1
        config.NUM_DECODER_LAYERS = 1
        config.MAX_PATH_LENGTH = 8 + 1
        config.MAX_NAME_PARTS = 5
        config.MAX_TARGET_PARTS = 6
        config.EMBEDDINGS_DROPOUT_KEEP_PROB = 0.75
        config.RNN_DROPOUT_KEEP_PROB = 0.5
        config.RANDOM_CONTEXTS = True
        config.USE_MOMENTUM = True
        return config

    def take_model_hyperparams_from(self, otherConfig):
        self.EMBEDDINGS_SIZE = otherConfig.EMBEDDINGS_SIZE
        self.NUM_ATTENTION_HEADS = otherConfig.NUM_ATTENTION_HEADS
        self.DECODER_SIZE = otherConfig.DECODER_SIZE
        self.NUM_DECODER_LAYERS = otherConfig.NUM_DECODER_LAYERS
        self.BIRNN = otherConfig.BIRNN
        if self.DATA_NUM_CONTEXTS <= 0:
            self.DATA_NUM_CONTEXTS = otherConfig.DATA_NUM_CONTEXTS

    def __init__(self, args):
        self.NUM_EPOCHS = 0
        self.SAVE_EVERY_EPOCHS = 0
        self.PATIENCE = 0
        self.BATCH_SIZE = 0
        self.TEST_BATCH_SIZE = 0
        self.READER_NUM_PARALLEL_BATCHES = 0
        self.SHUFFLE_BUFFER_SIZE = 0
        self.CSV_BUFFER_SIZE = None
        self.TRAIN_PATH = args.data_path
        self.TEST_PATH = args.test_path if args.test_path is not None else ''
        self.DATA_NUM_CONTEXTS = 0
        self.MAX_CONTEXTS = 0
        self.SUBTOKENS_VOCAB_MAX_SIZE = 0
        self.TARGET_VOCAB_MAX_SIZE = 0
        self.EMBEDDINGS_SIZE = 0
        self.NUM_ATTENTION_HEADS = 0
        self.DECODER_SIZE = 0
        self.NUM_ENCODER_LAYERS = 0
        self.NUM_DECODER_LAYERS = 0
        self.SAVE_PATH = args.save_path_prefix
        self.LOAD_PATH = args.load_path
        self.MAX_PATH_LENGTH = 0
        self.MAX_NAME_PARTS = 0
        self.MAX_TARGET_PARTS = 0
        self.EMBEDDINGS_DROPOUT_KEEP_PROB = 0
        self.RNN_DROPOUT_KEEP_PROB = 0
        self.RANDOM_CONTEXTS = True
        self.USE_MOMENTUM = True
        self.RELEASE = args.release

    @staticmethod
    def get_debug_config(args):
        config = Config(args)
        config.NUM_EPOCHS = 3000
        config.SAVE_EVERY_EPOCHS = 100
        config.PATIENCE = 200
        config.BATCH_SIZE = 7
        config.TEST_BATCH_SIZE = 7
        config.READER_NUM_PARALLEL_BATCHES = 1
        config.SHUFFLE_BUFFER_SIZE = 10
        config.CSV_BUFFER_SIZE = None
        config.MAX_CONTEXTS = 9
        config.SUBTOKENS_VOCAB_MAX_SIZE = 190000
        config.TARGET_VOCAB_MAX_SIZE = 27000
        config.EMBEDDINGS_SIZE = 32
        config.NUM_ATTENTION_HEADS = 1
        config.DECODER_SIZE = 11
        config.NUM_ENCODER_LAYERS = 1
        config.NUM_DECODER_LAYERS = 1
        config.MAX_PATH_LENGTH = 8 + 1
        config.MAX_NAME_PARTS = 5
        config.MAX_TARGET_PARTS = 6
        config.EMBEDDINGS_DROPOUT_KEEP_PROB = 1
        config.RNN_DROPOUT_KEEP_PROB = 1
        config.RANDOM_CONTEXTS = True
        config.USE_MOMENTUM = False
        return config
