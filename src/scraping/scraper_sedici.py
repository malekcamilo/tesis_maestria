from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import json, time
from collections import defaultdict
from typing import Dict, List, Union, Optional
from datetime import datetime
from urllib.parse import urlencode
import sys

class SEDICIScraper:
    def __init__(self, headless: bool = True, timeout: int = 10):
        self.base_url = "https://sedici.unlp.edu.ar"
        self.timeout = timeout
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        self.driver = webdriver.Chrome(options=options)
        self.driver.implicitly_wait(timeout)

    def close(self):
        self.driver.quit()

    def _extract_dublin_core(self) -> Dict[str, Union[str, List[str]]]:
        """Extrae metadatos Dublin Core de las etiquetas meta."""
        metas = self.driver.find_elements(
            By.XPATH,
            '//meta[starts-with(@name, "DC.") or starts-with(@name, "DCTERMS.")]',
        )
        # Usar defaultdict para agrupar valores
        temp_data = defaultdict(list)
        for m in metas:
            name = m.get_attribute("name")
            content = m.get_attribute("content")

            if not name or not content or not (content := content.strip()):
                continue

            clean = name.replace("DC.", "").replace("DCTERMS.", "")
            # Agregar solo valores únicos
            if content not in temp_data[clean]:
                temp_data[clean].append(content)

        # Convertir listas de un solo elemento a strings
        return {
            key: values[0] if len(values) == 1 else values
            for key, values in temp_data.items()
        }

    def _build_search_url(
        self, page: int, query: str = "", filtros: Optional[Dict[str, str]] = None
    ) -> str:
        params = {"rpp": 100, "page": page}
        if query:
            params["query"] = query.replace(" ", "+")
        filters = []
        if filtros:
            if "subject" in filtros:
                filters.append({"type": "subject", "value": filtros["subject"]})
            if "type" in filtros:
                filters.append({"type": "type", "value": filtros["type"]})
            start_date = filtros.get("date_from")
            end_date = filtros.get("date_to")
            if start_date and end_date:
                filters.append(
                    {"type": "dateIssued", "value": f"[{start_date} TO {end_date}]"}
                )
            elif start_date:
                filters.append({"type": "dateIssued", "value": start_date})
        for i, f in enumerate(filters):
            params[f"filtertype_{i}"] = f["type"]
            params[f"filter_relational_operator_{i}"] = "equals"
            params[f"filter_{i}"] = f["value"]
        return f"{self.base_url}/discover?{urlencode(params)}"

    def _get_item_links(self) -> List[str]:
        links = []
        elements = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/handle/"]')
        for e in elements:
            href = e.get_attribute("href")
            if (
                href
                and "/handle/" in href
                and all(
                    x not in href
                    for x in ["discover", "browse", "community-list", "submit"]
                )
            ):
                links.append(href)
        return list(set(links))

    def scrape_search(
        self,
        query: str = "",
        filtros: Optional[Dict[str, str]] = None,
        max_items: Optional[int] = None,
        start_page: int = 1,
    ) -> List[Dict]:
        results = []
        page = start_page
        visited = set()
        filtros_str = (
            ", ".join([f'{k}="{v}"' for k, v in filtros.items()]) if filtros else ""
        )
        max_items_str = str(max_items) if max_items is not None else "todos"
        print(
            f'Buscando: query="{query}", filtros={{{filtros_str}}} (max {max_items_str} items, comenzando desde la página {start_page})\n'
        )
        if max_items is None:
            max_items = sys.maxsize
        try:
            while len(results) < max_items:
                search_url = self._build_search_url(page, query, filtros)
                self.driver.get(search_url)
                try:
                    WebDriverWait(self.driver, self.timeout).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, 'a[href*="/handle/"]')
                        )
                    )
                except TimeoutException:
                    print(f"Timeout en página {page}")
                    break
                links = [l for l in self._get_item_links() if l not in visited]
                if not links:
                    print(f"No se encontraron más resultados en la página {page}")
                    break
                print(f"Página {page}: {len(links)} enlaces encontrados")
                for link in links:
                    if len(results) >= max_items:
                        break
                    visited.add(link)
                    self.driver.get(link)
                    metadata = self._extract_dublin_core()
                    results.append(metadata)
                    print(f"  Extraído: {metadata.get('title', 'sin título')}")
                    time.sleep(1.0)
                page += 1
                time.sleep(1.0)
        finally:
            self.close()
        return results

    def save_to_json(self, data: List[Dict], filename: str):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\nGuardado {len(data)} items en {filename}")

if __name__ == "__main__":
    # Ejemplo: Búsqueda con filtros
    # filtros = {
    #     'date_from': '2022',
    #     'date_to': '2024',
    #     'type': 'Tesis de doctorado',
    #     'subject': 'Informática'
    # }
    # resultados = scraper.scrape_search(query='Alguna consulta puntual',filtros=filtros)

    filtros = {
        "date_from": "2022",
        "date_to": "2024",
        "type": "Tesis",
        #'subject': 'Informática'
    }
    scraper = SEDICIScraper(headless=True)
    resultados = scraper.scrape_search(filtros=filtros, max_items=5)
    scraper.save_to_json(resultados, "sedici.json")
