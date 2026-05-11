from lore.builtins.tasks.ncbi.fetch_genome_reports import fetch_genome_reports_handler

def test_fetch_genome_reports(fake_ctx):    
    fetch_genome_reports_handler(fake_ctx, taxons=["Vibrio Cholerae"], search_terms=None, fetch_limit=1)
