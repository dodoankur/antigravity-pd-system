# 07. Monorepo Structure

We use Turborepo or Nx for managing the project.

```text
/root
  /apps
    /web           # Next.js frontend
    /admin-dash    # Internal monitoring tool
  /packages
    /ui            # Shared React components
    /core          # Shared JS utilities (math, types)
    /cv-models     # TFLite/WASM models and wrappers
  /services
    /scoring-api   # FastAPI scoring engine
    /auth-service  # Session management
  /docs            # Documentation suite
```
