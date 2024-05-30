from .base_options import BaseOptions


class TrainOptions(BaseOptions):

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument(
            "--phase", type=str, default="train", help="train, val, test, etc"
        )
        parser.add_argument(
            "--results_dir",
            type=str,
            default="samples_training",
            help="saves results here.",
        )
        parser.add_argument(
            "--save_latest_freq",
            type=int,
            default=5000,
            help="frequency of saving the latest results",
        )
        parser.add_argument(
            "--save_epoch_freq",
            type=int,
            default=1,
            help="frequency of saving checkpoints at the end of epochs",
        )
        parser.add_argument(
            "--save_by_iter",
            action="store_true",
            help="whether saves model by iteration",
        )
        parser.add_argument(
            "--print_freq",
            type=int,
            default=100,
            help="frequency of showing training results on console",
        )
        parser.add_argument(
            "--iter_loss", type=int, default=500, help="average loss for n iterations"
        )
        parser.add_argument(
            "--onnx_export",
            action="store_true",
            help="option to export trained model to ONNX format",
        )
<<<<<<< HEAD
        
=======
        parser.add_argument(
            "--transfer", action="store_true", help="build upon pre-trained model"
        )
>>>>>>> 2f4463fff66c40eb98ff6fa17c80c0caee775ac4
        parser.add_argument(
            "--unfreeze_layers_iters",
            type=int,
            default=0,
            help="rounds of gradually unfreezing layers for training",
        )
        parser.add_argument(
            "--unfreeze_interval",
            type=int,
            default=0,
            help="epoch interval between each unfreeze",
        )

        self.isTrain = True
        return parser
