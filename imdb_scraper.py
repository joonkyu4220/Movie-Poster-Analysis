import os
from os import dup
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen, urlretrieve

from time import sleep


from utils import try_mkdir, hmin_to_minutes, mk_to_int, DATA_ROOT_PATH

main_prefix = "https://www.imdb.com/title/"
main_suffix = "/"

next_class = 'lister-page-next'

title_datatestid = "hero-title-block__title"

releaseinfo_prefix = "/title/"
releaseinfo_suffix = "/releaseinfo?ref_=tt_ov_rdat"

score_prefix = "AggregateRatingButton__RatingScore"
len_score_prefix = len(score_prefix)

numscore_prefix = "AggregateRatingButton__TotalRatingAmount"
len_numscore_prefix = len(numscore_prefix)

genre_prefix = "/search/title/?genres="
len_genre_prefix = len(genre_prefix)

script_type_name = "application/ld+json"

keyword_suffix = "keywords?ref_=tt_stry_kw"
keyword_prefix = "/search/keyword?keywords="
len_keyword_prefix = len(keyword_prefix)

id_sep0 = "title"
id_sep1 = "tt"

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36',
           'Accept-Language' : 'en-US,en;q=0.9,ko-KR;q=0.8,ko;q=0.7'
          }

def retrieve_film_info(prefix, id, suffix):
    title = ""
    score = -1
    numscore = -1
    rated = ""
    year = -1
    time = -1
    genre = []
    keywords = []
    poster = ""

    url = prefix + id + suffix
    req = Request(url, headers=headers)
    sleep(0.5)
    html = urlopen(req)
    # soup = BeautifulSoup(html, "html.parser")
    soup = BeautifulSoup(html, "lxml")
    
    # html.text

    # TITLE RETRIEVAL
    for h1 in soup.findAll("h1"):
        if h1.get("data-testid") == title_datatestid:
            title = h1.string
            break

    # YEAR, RATED, TIME GENRE RETRIVAL
    for a in soup.findAll("a"):
        a_href = a.get("href")
        if not(a_href):
            continue
        elif a_href == (releaseinfo_prefix + id + releaseinfo_suffix):
            year = int(a.string)
            if a.parent and a.parent.next_sibling and a.parent.next_sibling.a:
                rated = a.parent.next_sibling.a
                time = rated.parent.next_sibling
                if time: time = hmin_to_minutes(time.string)
                rated = rated.string
            continue
        elif a_href[:len_genre_prefix] == genre_prefix:
            genre.append(a.string)
            continue
    
    # NUMSCORE RETRIEVAL
    for div in soup.findAll("div"):
        # # Popularity : Doesn't mean much, since new films are generally more popular.
        # div_datatestid = div.get("data-testid")
        # if div_datatestid == "hero-rating-bar__popularity__score":
        #     popularity = int(div.string)
        #     continue
        div_class = div.get("class")
        if (isinstance(div_class, list) and div_class[0][:len_numscore_prefix] == numscore_prefix):
            numscore = mk_to_int(div.string)
            # continue
            break
    
    # SCORE RETRIEVAL
    for span in soup.findAll("span"):
        span_class = span.get("class")
        if not(span_class):
            continue
        if (isinstance(span.get("class"), list) and span.get("class")[0][:len_score_prefix] == score_prefix):
            score = float(span.string)
            break

    # POSTER RETRIEVAL
    for script in soup.findAll("script"):
        script_type = script.get("type")
        if script_type == script_type_name:
            if "\"image\":\"" in script.text:
                poster = script.text.split("\"image\":\"")[1].split("\"")[0]


    html.close()


    # KEYWORDS RETRIEVAL
    url_kw = url + keyword_suffix
    req_kw = Request(url_kw, headers=headers)
    sleep(0.5)
    html_kw = urlopen(req_kw)
    # soup_kw = BeautifulSoup(html_kw, "html.parser")
    soup_kw = BeautifulSoup(html_kw, "lxml")
    for a in soup_kw.findAll("a"):
        a_href = a.get("href")
        if not(a_href):
            continue
        if a_href[:len_keyword_prefix] == keyword_prefix:
            keywords.append(a.string)
            continue
    html_kw.close()

    print(title)

    return {"id":id, "title":title, "score":score, "numscore":numscore, "rated":rated, "year":year, "time":time, "genre":genre, "keywords":keywords}, poster
    print("id: ", id)
    print("title: ", title)
    # print("popularity: ", popularity)
    print("score: ", score)
    print("numscore: ", numscore)
    print("rated: ", rated)
    print("year: ", year)
    print("time: ", time)
    print("genre: ", genre)
    print("keywords: ", keywords)
    print("\n")


def get_IDs_next(page):
    next_page = []
    ids = ['dummy']
    req = Request(page, headers=headers)
    sleep(0.5)
    html = urlopen(req)
    # soup = BeautifulSoup(html, "html.parser")
    soup = BeautifulSoup(html, "lxml")
    for link in soup.findAll('a'):
        href = link.get('href')
        if href is None:
            continue
        cls = link.get('class')
        if (cls is not None) and (cls[0] == next_class):
            next_page.append(href)
            continue
        href_split = link.get('href').split('/')
        if ((len(href_split) == 4) and (href_split[1] == id_sep0) and (href_split[2][:2] == id_sep1) and (href_split[2] != ids[-1])):
            ids.append(href_split[2])
            continue
    if len(next_page) == 0:
        page = None
    else:
        page = root_prefix + next_page[0]
    html.close()

    return ids[1:], page



if __name__ == "__main__":
    root_prefix = "https://www.imdb.com"
    
    first_page = root_prefix + "/search/title/?genres=comedy&explore=title_type,genres&title_type=movie&ref_=adv_explore_rhs"
    page = first_page
    os.chdir("D:")
    save_path = os.path.join(DATA_ROOT_PATH, "comedy")
    print("saving at:", save_path)
    try_mkdir(save_path)

    while (page):
        print("RUNNING {}".format(page))
        ids, page = get_IDs_next(page)
        for id in ids[:]:
            info, poster = retrieve_film_info(main_prefix, id, main_suffix)
            if info["id"] != "":
                txt_file = os.path.join(save_path, info["id"] + ".txt")
                img_file = os.path.join(save_path, info["id"] + ".jpg")
                if poster != "":
                    urlretrieve(poster, img_file)
                f_txt = open(txt_file, "w", encoding="UTF-8")
                for (k, v) in info.items():
                    f_txt.write(str(k))
                    f_txt.write("\t")
                    f_txt.write(str(v))
                    f_txt.write("\n")
                f_txt.close()