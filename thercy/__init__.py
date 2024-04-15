__thercy_phys_submodules__ = {'cycles', 'state', }

__all__ = [__thercy_phys_submodules__]


def __getattr__(attr):
    if attr == 'cycles':
        import thercy.cycles as cycles
        return cycles
    elif attr == 'state':
        import thercy.state as state
        return state
