# `app/modules` 

## Modular Solution For Functional Expansion

Each of the app/modules sub directories... needs a few things:

- `__init__.py` file
- `router.py` FastAPI Router file
- `<file>.py` File business logic

### Naming Modules For Continuity

Try to ensure that app/modules/<subdir name> is consistent with its route name and such.

With this we can route our logic to specific endpoints; whether from the app/... `main.py`, `api.py`, or whatever.