## Setup
1. Create the Conda environment:
   ```bash
   conda env create -f environment.yml -n name 
   ```

2. Activate the environment:
   ```bash
   conda activate name
   ```

3. Copy `.env.example` to `.env` and set your Elasticsearch credentials:
   ```
   ES_USER=your_username
   ES_PASSWORD=your_password
   ```

4. Ensure the `http_ca.crt` certificate file is in the root directory.

5. The application expects pre-built resources in the `resource/` directory:
   - `manual_indexer.pkl`
   - `pagerank_scores.pkl`

    If .pkl files are missing in resource/, run:
    ```bash
    python crawler.py
    python indexer.py
    ```

## Running the Application

```bash
python app.py
```
