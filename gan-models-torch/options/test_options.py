from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument(
            "--results_dir",
            type=str,
            default="samples_testing",
            help="saves results here.",
        )
        # parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.add_argument(
            "--phase", type=str, default="test", help="train, val, test, etc"
        )
        parser.add_argument(
            "--expand_dataset",
            action="store_true",
            help="use mask queue to artifically increase dataset variability",
        )
        parser.add_argument(
            "--shuffle",
            action="store_true",
            help="shuffle dataset to test out of order",
        )
        parser.set_defaults(load_size=parser.get_default("crop_size"))
        self.isTrain = False
        return parser
