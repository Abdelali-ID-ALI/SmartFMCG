"""Exercise the dashboard in Chromium while blocking external browser requests."""

from argparse import ArgumentParser
from pathlib import Path
from urllib.parse import urlsplit

from playwright.sync_api import expect, sync_playwright


def smoke(url: str, screenshots: Path | None = None) -> None:
    netloc = urlsplit(url).netloc
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        expect.set_options(timeout=40_000)

        def new_page():
            page = browser.new_page(viewport={"width": 1500, "height": 1240})
            page.route("**/*", lambda route: route.continue_()
                       if urlsplit(route.request.url).netloc == netloc else route.abort())
            return page

        page = new_page()
        page.goto(url, wait_until="domcontentloaded")

        def capture(name):
            if screenshots is not None:
                screenshots.mkdir(parents=True, exist_ok=True)
                expect(page.get_by_role("button", name="Stop", exact=True)).to_be_hidden()
                expect(page.locator('[data-testid="stException"], [data-testid="stAlert"][kind="error"]')).to_have_count(0)
                page.evaluate("document.fonts.ready")
                page.screenshot(path=screenshots / f"{name}.png", animations="disabled")

        def open_view(name, title=None):
            page.get_by_text(name, exact=True).click()
            expect(page.get_by_role("heading", name=title or name, exact=True)).to_be_visible()
            expect(page.locator('[data-testid="stException"]')).to_have_count(0)
            print(f"VIEW: {name}", flush=True)

        expect(page.get_by_role("heading", name="Distribution overview")).to_be_visible()
        open_view("Scenario configuration")
        page.get_by_role("button", name="Build scenario").click()
        expect(page.get_by_text("Scenario ready.", exact=False)).to_be_visible()
        expect(page.get_by_role("heading", name="Scenario data")).to_be_visible()
        expect(page.get_by_test_id("stDataFrame").first.locator("canvas").first).to_be_visible()
        capture("scenario_configuration")
        independent = new_page()
        independent.goto(url, wait_until="domcontentloaded")
        expect(independent.get_by_role("heading", name="Distribution overview")).to_be_visible()
        expect(independent.get_by_text("Loss savings vs B4", exact=True)).to_have_count(0)

        open_view("Route planning")
        page.get_by_role("button", name="Generate plans").click()
        expect(page.get_by_text("Archived OSRM geometry", exact=False).first).to_be_visible()
        expect(page.get_by_role("link", name="ODbL 1.0").first).to_have_attribute(
            "href", "https://www.openstreetmap.org/copyright")
        page.get_by_text("Map without external tiles or scripts", exact=True).click()
        expect(page.get_by_text("Road data © OpenStreetMap contributors", exact=False)).to_be_visible()
        expect(page.get_by_role("link", name="ODbL 1.0")).to_be_visible()
        expect(page.locator(".plotly").first).to_be_visible()
        capture("route_planning")

        open_view("Dispatch replay")
        page.get_by_role("button", name="Start paired simulation").click()
        expect(page.get_by_role("heading", name="Replay · B5 · Event-driven")).to_be_visible()
        page.get_by_role("button", name="Advance", exact=True).click()
        slider = page.get_by_role("slider")
        expect(slider).to_have_value("1")
        slider.press("End")
        expect(page.get_by_role("heading", name="Realized day comparison")).to_be_visible()
        page.get_by_role("tab", name="Known events").click()
        expect(page.get_by_text("Accrued loss", exact=True)).to_be_visible()
        capture("dispatch_replay")
        open_view("Overview", "Distribution overview")
        savings = page.get_by_text("Loss savings vs B4", exact=True)
        expect(savings).to_have_count(1)
        expect(savings).to_be_visible()
        expect(independent.get_by_text("Loss savings vs B4", exact=True)).to_have_count(0)
        capture("distribution_overview")
        print("FLOW: local caches, offline map, paired replay, isolated second session", flush=True)

        open_view("Travel-time prediction")
        expect(page.locator(".plotly").first).to_be_visible()
        page.get_by_role("tab", name="Residuals").click()
        expect(page.get_by_text("1,966 rows", exact=False)).to_be_visible()
        for label, count in (("Ordinary", 1891), ("Incidents", 75)):
            page.get_by_text(label, exact=True).click()
            result = page.get_by_text("rows · Residual", exact=False)
            expect(result).to_have_text(f"{count:,} rows · Residual = predicted − simulated actual")
            print(f"RESIDUAL FILTER {label}: {result.inner_text()}", flush=True)
        page.get_by_text("All", exact=True).click()
        expect(page.get_by_text("1,966 rows", exact=False)).to_be_visible()
        capture("travel_time_prediction")
        open_view("Learning policies")
        expect(page.locator(".plotly").first).to_be_visible()
        capture("learning_policies")
        open_view("Scientific benchmark")
        page.get_by_role("tab", name="Scalability").click()
        expect(page.get_by_text("Phase 10.1 · End-to-end OSRM replays", exact=False)).to_be_visible()
        expect(page.get_by_role("tabpanel", name="Scalability").locator(".plotly").first).to_be_visible()
        capture("scientific_benchmark")
        browser.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8512/")
    parser.add_argument("--screenshots", type=Path)
    args = parser.parse_args()
    smoke(args.url, args.screenshots)
