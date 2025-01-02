import arxiv

def get_arxiv_results(query, max_results):
    client = arxiv.Client()
    search = arxiv.Search(
        query=query, 
        max_results=max_results, 
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    results = client.results(search)
    return list(results)

def get_arxiv_message(result):
    summary = result.summary.replace('\n', ' ')
    authors = ', '.join([author.name for author in result.authors]) 
    message = (
        f"**Title:** {result.title}\n"
        f"**Authors:** {authors}\n"
        f"**Summary:** {summary}\n"
        f"**URL:** {result.entry_id}"
    )
    return message

