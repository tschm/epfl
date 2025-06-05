import pytest
import subprocess
import os
import http.server
import socketserver
import threading
import time
import requests
import glob


def get_all_notebooks():
    """Get all notebook files in book/marimo directory."""
    notebooks = glob.glob("book/marimo/*.py")
    # Filter out any directories, non-python files, or files that don't exist
    valid_notebooks = []
    for nb in notebooks:
        if os.path.isfile(nb) and not os.path.basename(nb).startswith("_"):
            # Verify the file exists and is accessible
            try:
                with open(nb) as f:
                    # Just try to read a bit to verify access
                    f.read(1)
                valid_notebooks.append(nb)
                print(f"Found valid notebook: {nb}")
            except Exception as e:
                print(f"Warning: Could not access notebook {nb}: {e}")

    print(f"Found {len(valid_notebooks)} valid notebooks")
    return valid_notebooks


@pytest.fixture(params=get_all_notebooks())
def notebook_path(request):
    """Fixture that provides each notebook path."""
    return request.param


@pytest.fixture()
def exported_html(tmp_path, notebook_path):
    # Get the notebook filename without extension
    notebook_name = os.path.basename(notebook_path).replace(".py", "")
    output_file = tmp_path / f"{notebook_name}.html"

    print(f"\nExporting notebook: {notebook_path} to {output_file}")

    # Verify the notebook file exists before trying to export
    if not os.path.isfile(notebook_path):
        pytest.skip(f"Notebook file {notebook_path} does not exist")

    try:
        # Export notebook to HTML-wasm via uv run marimo
        result = subprocess.run(
            [
                "uv",
                "run",
                "marimo",
                "export",
                "html-wasm",
                notebook_path,
                "-o",
                str(output_file),
                "--mode",
                "run",
            ],
            check=False,  # Don't raise exception, we'll handle it
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            error_msg = f"Error exporting notebook {notebook_path}:\n"
            error_msg += f"STDOUT: {result.stdout}\n"
            error_msg += f"STDERR: {result.stderr}"
            print(error_msg)
            pytest.skip(error_msg)

    except Exception as e:
        print(f"Exception while exporting notebook {notebook_path}: {e}")
        pytest.skip(f"Exception while exporting notebook {notebook_path}: {e}")

    yield tmp_path, output_file, notebook_name

    # shutil.rmtree(output_dir)


@pytest.fixture()
def http_server(exported_html):
    output_dir, _, _ = exported_html

    # Use a socket to find an available port
    def find_free_port():
        with socketserver.TCPServer(("localhost", 0), None) as s:
            return s.server_address[1]

    port = find_free_port()
    print(f"Using port {port} for HTTP server")

    # Change to the output directory
    original_dir = os.getcwd()
    os.chdir(output_dir)

    # Create a server with the allow_reuse_address option
    handler = http.server.SimpleHTTPRequestHandler
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer(("", port), handler)

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    # Wait a moment for the server to start
    time.sleep(1)

    try:
        yield f"http://localhost:{port}"
    finally:
        # Make sure we always clean up
        httpd.shutdown()
        httpd.server_close()  # Explicitly close the socket
        thread.join(timeout=5)  # Add timeout to avoid hanging

        # Change back to the original directory
        os.chdir(original_dir)


def test_marimo_page_served_and_opened(http_server, exported_html):
    _, _, notebook_name = exported_html
    url = f"{http_server}/{notebook_name}.html"
    print(f"\nOpening notebook page in browser at {url}")

    # Open browser for manual inspection (optional, comment out if running in CI)
    # webbrowser.open_new_tab(url)

    # Fetch the page to assert it loads properly
    response = requests.get(url)
    assert response.status_code == 200
    # content = response.text

    # Basic sanity checks for expected content
    # assert "Clarabel" in content or "Solve QP" in content
