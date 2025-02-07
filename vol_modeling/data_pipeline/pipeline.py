"""Main pipeline orchestration"""
class DataPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.loader = BBODataLoader()
        self.engineer = FeatureEngineer()
        self.universe = UniverseSelector()
    
    def process_single_day(self, date: str) -> None:
        """Process one day of data end-to-end"""
        pass
    
    def run_pipeline(self, start_date: str, end_date: str) -> None:
        """Run full pipeline for date range"""
        pass

# models/
