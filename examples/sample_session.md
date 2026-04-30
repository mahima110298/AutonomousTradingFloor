# Sample sessions to try

```bash
# Tech-heavy watchlist, single cycle
trading-floor trade -t "AAPL,MSFT,NVDA,GOOGL" -c 1

# Cross-sector, two cycles, fresh state
trading-floor trade -t "AAPL,JPM,XOM,JNJ" -c 2 --reset

# Inspect what each agent decided
trading-floor audit --kind decision -n 30

# Inspect just the executions
trading-floor audit --kind trade

# Export the full trail for offline analysis
trading-floor audit --limit 1000 > runs/audit.txt
```

The watchlist accepts any subset of:

```
AAPL  MSFT  GOOGL  NVDA  AMZN  TSLA  META  JPM  XOM  JNJ
```
