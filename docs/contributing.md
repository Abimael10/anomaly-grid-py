# Contributing

## Setup

```bash
git clone https://github.com/abimael10/anomaly-grid-py.git
cd anomaly-grid-py
./setup.sh
source venv/bin/activate
maturin develop
```

## Testing

```bash
pytest tests/
```

## Code Quality

```bash
black python/ tests/
ruff check python/ tests/
```

## Structure

```
src/           # Rust code
python/        # Python wrapper
tests/         # Tests
docs/          # Documentation
```

## Pull Requests

1. Fork repository
2. Create feature branch
3. Make changes
4. Run tests
5. Submit PR

## License

MIT