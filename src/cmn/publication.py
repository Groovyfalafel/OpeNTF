import json, os
from tqdm import tqdm
from time import time
from pathlib import Path

from cmn.author import Author
from cmn.team import Team

class Publication(Team):
    def __init__(self, id, authors, title, datetime, doc_type, venue, references, fos, keywords):
        super().__init__(id, authors, None, datetime, venue)
        self.title = title
        self.doc_type = doc_type
        self.location = venue
        self.references = references
        self.fos = fos
        self.keywords = keywords
        self.skills = self.set_skills()

        for author in self.members:
            author.teams.add(self.id)
            author.skills.update(set(self.skills))
        self.members_locations = [(str(venue), str(venue), str(venue))] * len(self.members)

    def set_skills(self):
        skills = set()
        for skill in self.fos:
            skills.add(skill["name"].replace(" ", "_").lower())
        return skills

    @staticmethod
    def read_data(datapath, output, index, filter, settings):
        st = time()
        # Convert paths to Path objects for better path handling
        output_path = Path(output)
        teamsvecs_path = output_path / "teamsvecs.pkl"
        indexes_path = output_path / "indexes.pkl"
        
        try:
            print(f"Attempting to load pickle files from {output_path}")
            return super(Publication, Publication).load_data(str(output_path), index)
        except (FileNotFoundError, EOFError) as e:
            print(f"Pickles not found! Reading raw data from {datapath} (progress in bytes) ...")
            teams = {}
            candidates = {}

            datapath = Path(datapath)
            if not datapath.is_file():
                raise FileNotFoundError(f"Data file not found: {datapath}")

            with tqdm(total=os.path.getsize(datapath)) as pbar, open(datapath, "r", encoding='utf-8') as jf:
                for line in jf:
                    try:
                        if not line: break
                        pbar.update(len(line))
                        jsonline = json.loads(line.lower().lstrip(","))
                        # ... rest of your processing code ...
                        
                        id = jsonline['id']
                        title = jsonline['title']
                        year = jsonline['year']
                        type = jsonline['doc_type']
                        venue = jsonline['venue'] if 'venue' in jsonline.keys() else None
                        references = jsonline['references'] if 'references' in jsonline.keys() else []
                        keywords = jsonline['keywords'] if 'keywords' in jsonline.keys() else []

                        try: 
                            fos = jsonline['fos']
                        except: 
                            continue

                        try: 
                            authors = jsonline['authors']
                        except: 
                            continue

                        members = []
                        for author in authors:
                            member_id = author['id']
                            member_name = author['name'].replace(" ", "_")
                            member_org = author['org'].replace(" ", "_") if 'org' in author else ""
                            if (idname := f'{member_id}_{member_name}') not in candidates:
                                candidates[idname] = Author(member_id, member_name, member_org)
                            members.append(candidates[idname])
                        team = Publication(id, members, title, year, type, venue, references, fos, keywords)
                        teams[team.id] = team
                        if 'nrow' in settings['domain']['dblp'].keys() and len(teams) > settings['domain']['dblp']['nrow']: 
                            break

                    except json.JSONDecodeError as e:
                        print(f'JSONDecodeError: There has been error in loading json line `{line}`!\n{e}')
                        continue
                    except Exception as e:
                        raise e

            print(f"Processed {len(teams)} teams")
            print(f"Output directory: {output_path}")
            return super(Publication, Publication).read_data(teams, str(output_path), filter, settings)
        except Exception as e: 
            print(f"Error reading data: {str(e)}")
            raise e