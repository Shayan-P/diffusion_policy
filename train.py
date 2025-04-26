"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
import code
import importlib
import time

# Add autoreload functionality
def enable_autoreload():
    from importlib import reload
    import builtins
    
    # Keep track of module load times
    module_load_times = {}
    original_import = builtins.__import__
    
    def autoreload_import(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)
        
        # Only reload user modules (not built-in or third-party modules)
        if (hasattr(module, '__file__') and 
            'site-packages' not in str(module.__file__) and 
            'dist-packages' not in str(module.__file__) and
            'python3' not in str(module.__file__)):
            
            file_path = module.__file__
            last_modified = module_load_times.get(file_path, 0)
            current_time = time.time()
            
            try:
                if current_time - last_modified > 1:  # Reload if more than 1 second has passed
                    module = importlib.reload(module)
                    module_load_times[file_path] = current_time
            except:
                pass  # Silently fail if reload fails
                
        return module
    
    builtins.__import__ = autoreload_import


# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    def run():
        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg)
        workspace.run()

    if cfg.get('interactive', False):
        enable_autoreload()  # Enable autoreload before interactive mode
        code.interact(local=locals())
    else:
        run()

if __name__ == "__main__":
    main()
