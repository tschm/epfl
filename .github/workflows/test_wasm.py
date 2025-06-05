import pytest
import subprocess
import os
import http.server
import socketserver
import threading
import time
import requests


@pytest.fixture(scope="module")
def exported_html(tmp_path):
    notebook = "book/marimo/ConditionalValueAtRisk.py"
    # output_dir = tempfile.mkdtemp(prefix="marimo_run_")
    output_file = tmp_path / "ConditionalValueAtRisk.html"

    # Export notebook to HTML-wasm via uv run marimo
    subprocess.run(
        [
            "uv",
            "run",
            "marimo",
            "export",
            "html-wasm",
            notebook,
            "-o",
            str(output_file),
            "--mode",
            "run",
        ],
        check=True,
    )

    yield tmp_path, output_file

    # shutil.rmtree(output_dir)


@pytest.fixture(scope="module")
def http_server(exported_html):
    output_dir, _ = exported_html
    port = 8000

    os.chdir(output_dir)

    handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", port), handler)

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    # Wait a moment for the server to start
    time.sleep(1)

    yield f"http://localhost:{port}"

    httpd.shutdown()
    thread.join()


def test_marimo_page_served_and_opened(http_server, exported_html):
    url = f"{http_server}/ConditionalValueAtRisk.html"
    print(f"\nOpening notebook page in browser at {url}")

    # Open browser for manual inspection (optional, comment out if running in CI)
    # webbrowser.open_new_tab(url)

    # Fetch the page to assert it loads properly
    response = requests.get(url)
    assert response.status_code == 200
    # content = response.text

    # Basic sanity checks for expected content
    # assert "Clarabel" in content or "Solve QP" in content
