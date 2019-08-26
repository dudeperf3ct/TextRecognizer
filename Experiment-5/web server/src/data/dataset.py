"""
Dataset class to be extended by dataset-specific classes.
"""

class Dataset:
    """Simple abstract class for datasets."""
    def download(self):
        raise NotImplementedError("This is an abstract class. Method not Implement yet!")

    def load_data(self):
        raise NotImplementedError("This is an abstract class. Method not Implement yet!")