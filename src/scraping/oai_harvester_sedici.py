import requests
import xml.etree.ElementTree as ET
import json
from datetime import datetime
import time

class OAIPMHHarvester:
    def __init__(self, base_url):
        self.base_url = base_url
        self.namespaces = {
            'oai': 'http://www.openarchives.org/OAI/2.0/',
            'oai_dc': 'http://www.openarchives.org/OAI/2.0/oai_dc/',
            'dc': 'http://purl.org/dc/elements/1.1/'
        }
    
    def list_metadata_formats(self):
        params = {'verb': 'ListMetadataFormats'}
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            formats = []
            for fmt in root.findall('.//oai:metadataFormat', self.namespaces):
                prefix = fmt.find('oai:metadataPrefix', self.namespaces)
                if prefix is not None:
                    formats.append(prefix.text)
            return formats
        return []
    
    def harvest_records(self, metadata_prefix='oai_dc', set_spec=None, from_date=None, until_date=None):
        records = []
        params = {
            'verb': 'ListRecords',
            'metadataPrefix': metadata_prefix
        }
        if set_spec:
            params['set'] = set_spec
        if from_date:
            params['from'] = from_date
        if until_date:
            params['until'] = until_date
        
        resumption_token = None
        request_count = 0
        
        while True:
            try:
                if resumption_token:
                    params = {
                        'verb': 'ListRecords',
                        'resumptionToken': resumption_token
                    }
                
                print(f"Solicitando registros (petición #{request_count + 1})...")
                response = requests.get(self.base_url, params=params, timeout=60)
                
                if response.status_code != 200:
                    print(f"Error en la petición: {response.status_code}")
                    break
                
                root = ET.fromstring(response.content)
                
                # Verificar si hay errores
                error = root.find('.//oai:error', self.namespaces)
                if error is not None:
                    print(f"Error OAI-PMH: {error.get('code')} - {error.text}")
                    break
                
                # Extraer registros
                for record in root.findall('.//oai:record', self.namespaces):
                    metadata = self.extract_metadata(record)
                    if metadata:
                        records.append(metadata)
                
                print(f"Registros extraídos hasta ahora: {len(records)}")
                
                # Buscar resumption token
                resumption = root.find('.//oai:resumptionToken', self.namespaces)
                if resumption is not None and resumption.text:
                    resumption_token = resumption.text
                    request_count += 1
                    time.sleep(1)  # Pausa para no sobrecargar el servidor
                else:
                    break
                    
            except Exception as e:
                print(f"Error durante la cosecha: {str(e)}")
                break
        
        print(f"\nTotal de registros cosechados: {len(records)}")
        return records
    
    def extract_metadata(self, record):
        """
        Extrae los metadatos de un registro individual
        
        Args:
            record: Elemento XML del registro
        
        Returns:
            Diccionario con los metadatos del registro
        """
        metadata = {}
        
        # Extraer identificador
        identifier = record.find('.//oai:identifier', self.namespaces)
        metadata['identifier'] = identifier.text if identifier is not None else ''
        
        # Extraer status (si el registro está borrado)
        header = record.find('.//oai:header', self.namespaces)
        metadata['status'] = header.get('status', 'active') if header is not None else 'active'
        
        # Extraer metadatos Dublin Core
        dc_metadata = record.find('.//oai_dc:dc', self.namespaces)
        
        if dc_metadata is not None:
            # Campos Dublin Core comunes
            dc_fields = [
                'title', 'creator', 'subject', 'description', 'publisher',
                'contributor', 'date', 'type', 'format', 'identifier',
                'source', 'language', 'relation', 'coverage', 'rights'
            ]
            
            for field in dc_fields:
                elements = dc_metadata.findall(f'dc:{field}', self.namespaces)
                if elements:
                    # Extraer valores y eliminar duplicados manteniendo el orden
                    values = []
                    seen = set()
                    for elem in elements:
                        if elem.text and elem.text not in seen:
                            values.append(elem.text)
                            seen.add(elem.text)
                    
                    if values:
                        # Si hay un solo valor, guardar como string, si hay múltiples como lista
                        metadata[field] = values[0] if len(values) == 1 else values
                    else:
                        metadata[field] = ''
                else:
                    metadata[field] = ''
        
        return metadata
    
    def save_to_json(self, records, output_file='oai_metadata.json', indent=2):
        """
        Guarda los registros en un archivo JSON
        
        Args:
            records: Lista de diccionarios con metadatos
            output_file: Nombre del archivo JSON de salida
            indent: Nivel de indentación (default: 2, usar None para compacto)
        """
        if not records:
            print("No hay registros para guardar")
            return
        
        # Escribir JSON directamente con los registros
        with open(output_file, 'w', encoding='utf-8') as jsonfile:
            json.dump(records, jsonfile, ensure_ascii=False, indent=indent)
        
        print(f"\nDatos guardados en: {output_file}")
        print(f"Total de registros: {len(records)}")

if __name__ == "__main__":
    # Configurar la URL del repositorio OAI-PMH
    BASE_URL = "https://sedici.unlp.edu.ar/oai/request"
    
    # Crear el harvester
    harvester = OAIPMHHarvester(BASE_URL)
    
    # Listar formatos disponibles (opcional)
    print("Formatos de metadatos disponibles:")
    formats = harvester.list_metadata_formats()
    for fmt in formats:
        print(f"  - {fmt}")
    print()
    
    # Cosechar registros
    records = harvester.harvest_records(
        metadata_prefix='oai_dc',  # Cambiar según necesites
        # set_spec='publicaciones',  # Opcional: especificar un set
         from_date='2025-10-10',    # Opcional: desde una fecha
        #until_date='2025-10-10'    # Opcional: hasta una fecha
    )
    
    # Guardar en JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'oai_metadata_{timestamp}.json'
    harvester.save_to_json(records, output_file)