
####==================Extract information from spec document==========================================================================###

def spec_extract(path):
    
    """
    Args:
        :path: the absolute path to the specs
        
    Returns:
        :spec_df: the dataframe for all processed specs
        :corrupt_files: the corrupt files
    """
    
    # !pip3 install mammoth
    
    import os
    import mammoth
    import pandas as pd
    from bs4 import BeautifulSoup
    import re
    import datetime
    
    
    
    Rawtext, Jobtitle, Location, Education, Skills,\
    Experience, spec_identifier, corrupt_files = [], [], [], [], [], [], [], []
    
    
    for sample_spec in os.listdir(path):
    
        try:
            
            ## Attach an identifier for the document
            
            spec_identifier.append(sample_spec.split('.')[0])
            
            
            ### Open and process the document

            with open(path+sample_spec, "rb") as file:
                result = mammoth.convert_to_html(file)
                spec_html = result.value


            spec_soup = BeautifulSoup(spec_html)
            spec_text = spec_soup.get_text('\n')

            spec_split_lines = spec_text.splitlines()

            Rawtext.append(spec_text)


            ## Extract jobtitle
            job_name = ""
            for line in spec_split_lines:
                if re.search("Job Title", line):
                    start = line.find(":")
                    job_name += line[start + 2:]
                    #print(job_name)
                    break

            Jobtitle.append(job_name)


            ## Extract loaction
            area_name = ""
            for line in spec_split_lines:
                if re.search("Area", line):
                    start = line.find(":")
                    area_name += line[start + 2:]
                    #print(area_name)
                    break

            province_name = ""
            for line in spec_split_lines:
                if re.search("Province", line):
                    start = line.find(":")
                    province_name += line[start + 2:]
                    #print(province_name)
                    break

            location = area_name + ", " + province_name

            Location.append(location)


            ## Extract education
            possible_education_keywords = ["Master", "diploma", "degree", "grade 12"]

            for line in spec_split_lines:
                for edu_word in possible_education_keywords:
                    if re.search(edu_word, line, re.IGNORECASE):
                        Education.append(line)
                        #print(line)
                        break


            ## Extract skills
            skill_section_dividers = ["What does it take", "Technology stack", "Skills", "Requirements", "Competencies", "Qualifications"]

            for line in spec_split_lines:
                for section in skill_section_dividers:
                    if re.search(section, line):
                        #print(line)
                        #print(spec_split_lines.index(line))
                        section_line = line
                        break

            skills = []
            section_index = spec_split_lines.index(section_line)
            #print(section_index)
            for line in spec_split_lines[section_index + 1:]:
                skills.append(line)

            Skills.append(skills)


            ## Extract experience
            experience = []

            for line in spec_split_lines:
                if re.search("experience", line, re.IGNORECASE):
                    if re.search("year|years", line, re.IGNORECASE):
                        experience.append(line)

            Experience.append(experience)
            
        except Exception:
            
            corrupt_files.append(sample_spec.split('.')[0])
            
            continue

        
    spec_dict = {"spec_identifier": spec_identifier,
                "rawtext": Rawtext,
                "jobtitle": Jobtitle,
                "area": Location,
                "education": Education,
                "skills": Skills,
                "experience": Experience}
    
    spec_df = pd.DataFrame(dict([(key, pd.Series(val)) for key, val in spec_dict.items()])).fillna("")

    spec_df["process_date"] = datetime.datetime.now()
    
    spec_df = spec_df[spec_df.rawtext!=""]

    return spec_df, corrupt_files

####====================================================================================================================================





####=======================Function to create url search strings for recruiters' job specs===============================================

def urlsearchstring(spec_df):
    
    """
    Args:
        spec_df: is a spec dataframe
            
    Returns:
        A list of LinkedIn and GitHub site search strings to be used for querying canduidates on linkedin
        and github.
    """
    ### Load the required packages
    import urllib

    ## Create a frame from the listed jobtitles in the spec dataframe

    file = spec_df['jobtitle']
    
    ## Create a frame for the listed areas in the spec dataframe

    #area = []
    #for i in range(len(spec_df)):
        #area.append(spec_df.area[i][0])
    #area = pd.DataFrame(area)[0]
    area = spec_df['area']
    
    
    # Add the linkedin site search to the search string with restriction to SA
    file_l = "site:za.linkedin.com/in " + file + " " + area

    #### Parse the linkedin search string as a url and add to the URL search term (string).
    file_l = file_l.apply(lambda x: urllib.parse.quote_plus(str(x)))
    file_l1 = "https://www.google.com/search?q=" + file_l + ' &start=0'
    file_l2 = "https://www.google.com/search?q=" + file_l + ' &start=10'
    file_l3 = "https://www.google.com/search?q=" + file_l + ' &start=20'
    
    # Add the github site search to the search string without restriction to SA
    file_g = "github.com " + file + " followers " + area
    
    #### Parse the github search string as a url and add to the URL search term (string).
    file_g = file_g.apply(lambda x: urllib.parse.quote_plus(str(x)))
    file_g1 = "https://www.google.com/search?q=" + file_g + ' &start=0'
    file_g2 = "https://www.google.com/search?q=" + file_g + ' &start=10'
    
   
    return file_l1, file_l2, file_l3, file_g1, file_g2


####======================================================================================================================================


###=================================== Function to get URLs of Google Search Results (GSR).=====================================

def gsr_urls_scrapper(url, domain):
    
    """
    Return the source code for the provided URL.
        Args:
            url (string): URL of the page to scrape.
            searched_domain (tuple): domain one is searching result from.
        Returns:
            links (list): list of urls of Google Search Results (GSR).
    """
    ### Load the required packages
    import numpy as np
    import requests
    from requests_html import HTMLSession

    try:
        session = HTMLSession()
        response = session.get(url)

        links = list(response.html.absolute_links)

        output = []
        for link in links[:]:
            if domain in link:
                output.append(link)

    except requests.exceptions.RequestException:
         output = []

    return np.unique(output).tolist()
###===============================================================================================================================


###======================================= Function to get LinkedIn GSR clean URLs===============================================

def generate_gsr_urls(spec_df, qterm_func=urlsearchstring, gsr_scrapper_func=gsr_urls_scrapper):
    
    """
    Args:
        df: input dataframe
        gsr_urls_scrapper: function that scrapes the html source containing url links.
        linkedin_query_term: function that generates the google search terms.
    Returns:
            dataframe with linkedIn  and Github urls for a given spec dataframe.

    Usage:
        generate_gsr_urls(spec_df)
    """
    
    ### Load required packages.
    import pandas as pd
    import numpy as np

    ### Get the the searched term URLs
    linkedin_gs_terms1, linkedin_gs_terms2, linkedin_gs_terms3,\
    github_gs_terms1, github_gs_terms2 = qterm_func(spec_df)

    out_dict_l = [] 
    out_dict_g = []

    for spec_identifier, lgs_term1, lgs_term2, lgs_term3, ggs_term1, ggs_term2, spec_jobtitle\
    in zip(spec_df.spec_identifier, linkedin_gs_terms1, linkedin_gs_terms2,\
           linkedin_gs_terms3, github_gs_terms1, github_gs_terms2, spec_df.jobtitle):
        
	# LinkedIn urls
        lgsr_urls1 = gsr_scrapper_func(lgs_term1, "za.linkedin.com")
        lgsr_urls2 = gsr_scrapper_func(lgs_term2, "za.linkedin.com")
        lgsr_urls3 = gsr_scrapper_func(lgs_term3, "za.linkedin.com")
        
        url_identifier1 = [str(url).split("?")[0] for url in lgsr_urls1]
        url_identifier2 = [str(url).split("?")[0] for url in lgsr_urls2]
        url_identifier3 = [str(url).split("?")[0] for url in lgsr_urls3]
        url_identifier = url_identifier1 + url_identifier2 + url_identifier3
        

        out_dict_i = {"spec_identifier": [spec_identifier for i in range(len(url_identifier))],
                     "url_identifier": np.unique(url_identifier).tolist(),
                     "spec_jobtitle": [spec_jobtitle for i in range(len(url_identifier))]}

        out_dict_l.append(pd.DataFrame(dict([(key, pd.Series(val)) for key, val in out_dict_i.items()])).fillna(""))
    
	## Github urls
        ggsr_urls1 = gsr_scrapper_func(ggs_term1, "github.com")
        ggsr_urls2 = gsr_scrapper_func(ggs_term2, "github.com")
        urls_identifier1 = [str(url).split("?")[0] for url in ggsr_urls1]
        urls_identifier2 = [str(url).split("?")[0] for url in ggsr_urls2]
        urls_identifier = urls_identifier1 + urls_identifier2
        
        out_dict_g1 = {"spec_identifier": [spec_identifier for i in range(len(urls_identifier))],
                     "url_identifier": np.unique(urls_identifier).tolist(),
                     "spec_jobtitle": [spec_jobtitle for i in range(len(urls_identifier))]}

        out_dict_g.append(pd.DataFrame(dict([(key, pd.Series(val)) for key, val in out_dict_g1.items()])).fillna(""))


    ## Concatenate and process LinkedIn urls    
    out_df_l = pd.concat(out_dict_l, ignore_index=True) 
    out_df_l['url_identifier'] = out_df_l['url_identifier'].apply(lambda x:x if "linkedin.com/in" in x else "")
    out_df_l = out_df_l[out_df_l['url_identifier']!=""].reset_index(drop=True)


    ## Concatenate and process Github urls
    out_df_g = pd.concat(out_dict_g, ignore_index=True)
    out_df_g['url_identifier'] = out_df_g['url_identifier'].apply(lambda x:x if "https://github.com" in x else "")
    out_df_g = out_df_g[out_df_g['url_identifier']!=""].reset_index(drop=True)
    out_df_g['username'] = out_df_g.url_identifier.apply(lambda x: x.split("/")[3] if "https://github.com" in x else "")

    
    return out_df_l, out_df_g
###=================================================================================================================================



#####===================================Extractor of a given LinkedIn or GitHub profile===========================================## 
def profile_extractor(url):
    
    """
    Description:
    ------------
        - url: [str], the url of a LinkedIn/GitHub profile page.
        - The function returns the HTML code of the profile page.
    """
    from bs4 import BeautifulSoup
    from proxycrawl.proxycrawl_api import ProxyCrawlAPI

    api = ProxyCrawlAPI({"token": "4_41ALAy5mxlBOKwLEFtmQ"})

    response_profile = api.get(url)
    profile_data = ""
    if response_profile["status_code"] == 200:
        profile_data = BeautifulSoup(response_profile["body"], "html.parser")

    return profile_data
#####================================================================================================================================##



def linkedin_extractor(linkedin_profile_url):
    
    import requests

    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    #linkedin_profile_url = 'https://za.linkedin.com/in/craig-matthee-10b70825'  #linkedIn_urls.url_id[1]
    api_key = 'a3673d04-e1f8-486d-9ac9-7eb013061462'
    header_dic = {'Authorization': 'Bearer ' + api_key}

    response = requests.get(api_endpoint,
                          params={'url': linkedin_profile_url},
                          headers=header_dic)

    return response.json()




def extr_nickname(html_code):
    
    '''extracting username/nickname'''
    
    nickname = html_code.find_all(class_='p-nickname')
    if nickname:
        name = nickname[0].text.strip()
    else:
        name = ''
    return name
    
    
def extr_job(html_code):
    
    '''extracting job title'''
    
    job = html_code.find_all(class_='p-note')
    if job:
        job = job[0].text.strip()
        job = job.split('@')[0] # removing any org present
        if ',' in job: # if multiple present
            job = job.split(',')
        job_title = job
    else:
        job_title = ''
    return job_title
    
    
def extr_org(html_code):
    
    '''extracting organization individual works at'''
    
    org = html_code.find_all(class_='p-org')
    if org:
        org_name = org[0].text.strip()
    else:
        org_name = ''
    return org_name
    

def extr_location(html_code):
    '''extracting location/area'''
    location = html_code.find_all(class_='p-label')
    if location:
        area = location[0].text.strip()
    else:
        area = ''
    return area
        

def extr_years(html_code):
    
    '''calculating years of experience'''
    
    # extract earliest year on github to calculate
    year_start = html_code.find_all(class_='js-year-link')[-1].text
    current_year = date.today().year
    exp_yrs = int(current_year) - int(year_start)
    return exp_yrs
    

def extr_exp(html_code):
    
    '''extracting experience (using markdown body & pinned project descriptions if present)'''
    
    exp_list = []
    
    body_descr = html_code.find_all(class_='markdown-body')
    pinned_descr = html_code.find_all(class_='pinned-item-desc')
    
    if body_descr or pinned_descr:
        for point in body_descr:
            exp_list.append(point.text.strip())
        for point in pinned_descr:
            exp_list.append(point.text.strip())
    else:
        exp_list = ''
        
    return exp_list



def extr_skills(url):
    
    '''extracting coding languages used'''
    
    # new url with all repositories
    repo_url = url + '?tab=repositories'
    # convert to html code
    html_code = profile_extractor(repo_url)
    
    # singling out dropdown menu of all used languages to extract
    language_html = html_code.find(id="language-options")
    
    # append all present languages to list
    language_lst = []
    for language in language_html.find_all(class_="text-normal"):
        if language.text != 'All':
            language_lst.append(language.text)
    
    return language_lst



# combining extraction function outputs into a single dictionary

def github_profile_extractor(github_df, profile_extractor=profile_extractor, extr_nickname=extr_nickname,\
                     extr_job=extr_job, extr_org=extr_org, extr_location= extr_location,\
                     extr_years=extr_years, extr_exp=extr_exp, extr_skills=extr_skills):
    
    
    ''' 
    Args:
        takes github url extracts desired details
    '''
    
    import pandas as pd
    
    github_dataframe = []
    
    for i, j, k in zip(github_df.spec_identifier, github_df.url_identifier, github_df.spec_jobtitle):
        
        # extract html code
        html_code = profile_extractor(j)

        # empty dict to be filled
        spec_dict = {}

        # extracting elements & adding to dict
        
        spec_dict['spec_identifier'] = i
        spec_dict['spec_jobtitle'] = k
        
        try:
            name = html_code.find_all(class_='p-name')[0].text.strip()
            spec_dict['name'] = name
        except:
            spec_dict['name'] = ""
            
        try:
            spec_dict['username'] = extr_nickname(html_code)
        except:
            spec_dict['username'] = ""
            
        try:
            spec_dict['profile_jobtitle'] = extr_job(html_code)
        except:
            spec_dict['profile_jobtitle'] = ""
        
        try:
            spec_dict['org'] = extr_org(html_code)
        except:
            spec_dict['org'] = ""
            
        try:
            spec_dict['location'] = extr_location(html_code)
        except:
            spec_dict['location'] = ""
            
        try:
            spec_dict['years of exp'] = extr_years(html_code)
        except:
            spec_dict['years of exp'] = 0
            
        try:
            spec_dict['experience'] = extr_exp(html_code)
        except:
            spec_dict['experience'] = ""
            
        try:
            spec_dict['skills'] = extr_skills(url)
        except:
            pec_dict['skills'] = ""
        
        github_dataframe.append(pd.DataFrame(spec_dict, index=range(1)))
        
   
    # skills - have yet to extract languages
        
    return pd.concat(github_dataframe, ignore_index=True)
    

    
def linkedin_profile_extractor(linkedin_df, linkedin_extractor=linkedin_extractor):
    
    ''' extracting desired information from LinkedIn Extractor dictionary'''
    
    import pandas as pd                        
    
    linkedin_dataframe = []
    
    for i, j, k in zip(linkedin_df.spec_identifier, linkedin_df.url_identifier, linkedin_df.spec_jobtitle):
        
        
        candidate_dict = linkedin_extractor(j)
        
        # generating dictionary of all the specs
        clean_dict = {}

        # Spec identifier
        clean_dict['spec_identifier'] = i
        
        # Spec jobtitle
        clean_dict['spec_jobtitle'] = k
        
        # name
        clean_dict['name'] = candidate_dict['full_name']

        # job title
        clean_dict['job title'] = candidate_dict['occupation']

        # area
        city = candidate_dict['city']
        state = candidate_dict['state']
        country = candidate_dict['country']
        clean_dict['area'] = str(f'{city}, {state}, {country}')

        # education
        degree_list = []
        for edu in candidate_dict['education']:
            degree_list.append(edu['degree_name'])
        clean_dict['education'] = degree_list

        # experience
        job_list = []
        for job in candidate_dict['experiences']:
            job_list.append(job['title'])
        clean_dict['experience'] = job_list

        # skills / experience description
        # summary & descriptions of experiences will include skills & experience
            # skills section of LinkedIn Profile not included in scrapper code
        description_list = []
        description_list.append(candidate_dict['summary'])
        for job in candidate_dict['experiences']:
            description_list.append(job['description'])
        clean_dict['skills'] = description_list
        
        
        linkedin_dataframe.append(pd.DataFrame(clean_dict, index=range(1)))
        

        # note: industry hard to categorize
    
    return pd.concat(linkedin_dataframe, ignore_index=True)
