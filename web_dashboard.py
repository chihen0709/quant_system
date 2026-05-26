from __future__ import annotations

import os

import uvicorn

from quant_web import app


def main() -> None:
    host = os.environ.get('WEB_HOST', '0.0.0.0')
    port = int(os.environ.get('WEB_PORT', '8000'))
    uvicorn.run('quant_web:app', host=host, port=port, reload=False)


if __name__ == '__main__':
    main()
