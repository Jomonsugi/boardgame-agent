import warnings

# Suppress transformers v5 deprecation warnings for legacy image processor path aliases.
# These are triggered by docling's transitive import of transformers and are harmless.
warnings.filterwarnings("ignore", message="Accessing `__path__`", module="transformers")
warnings.filterwarnings("ignore", message="resource_tracker:", module="multiprocessing")
