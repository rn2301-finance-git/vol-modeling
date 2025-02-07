"""Daily trading universe management"""
class UniverseSelector:
    def select_daily_universe(self, date: str, 
                            lookback_days: int = 10) -> List[str]:
        """Select top N stocks by turnover"""
        pass
    
    def get_active_symbols(self, date: str) -> Set[str]:
        """Get set of active symbols for given date"""
        pass
