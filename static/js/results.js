const urlParams = new URLSearchParams(window.location.search);
const query = urlParams.get('query');

if (query) {
    document.getElementById('query-input').value = query;
    fetchResults('/search_es_pr', query, 'es-meta', 'es-results');
    fetchResults('/search_manual_pr', query, 'manual-meta', 'manual-results');
}

function fetchResults(endpoint, q, metaId, resultsId) {
    fetch(`${endpoint}?query=${encodeURIComponent(q)}`)
        .then(response => response.json())
        .then(data => {
            const metaEl = document.getElementById(metaId);
            const resultsEl = document.getElementById(resultsId);

            metaEl.textContent = `About ${data.total_hit} results (${data.elapse.toFixed(4)} seconds)`;

            let html = '';
            data.results.forEach(item => {
                html += `
                    <div class="result-item">
                        <div class="result-url">${item.url || 'No URL available'}</div>
                        <a href="${item.url || '#'}" class="result-title">${item.title || 'Untitled'}</a>
                        <div class="result-snippet">${item.text}</div>
                    </div>
                `;
            });
            resultsEl.innerHTML = html;
        })
        .catch(error => {
            document.getElementById(metaId).textContent = "Error loading results.";
        });
}