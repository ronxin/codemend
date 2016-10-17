import re

from codemend import relative_path

with open(relative_path('models/output/mpl_so_titles.txt')) as reader:
  id_titles = [x.split('\t') for x in reader.read().decode('utf-8').split('\n')]

print 'There are %d threads in total'%len(id_titles)

count = 0
how_prefixes = ['how can i', 'how do i', 'how do you', 'how would one', 'how to',
                'how should i', 'how do we', 'how is it possible to', 'how does one',
                'how i can', 'how could i', 'how can one', 'how we can', 'how can you',
                'do i have to',
                'how i', 'how do', 'how would you', 'how would i', 'how would', 'how should',
                'how can', 'how are', 'how / where to', 'how']  # order is important
goals = []
for x in id_titles:
  if not len(x) == 2: continue
  id_, title = x
  title = title.lower()
  if title.startswith('how'):
    for hp in how_prefixes:
      title = title.replace(hp, '').strip()
    goals.append((id_,title))

print 'There are %d how-questions'%len(goals)

initial_words = set(map(lambda x: x[1].split()[0], goals))

# Second pass. Extract those titles that start with initial words of these
# goals extracted from the how-to questions.
for x in id_titles:
  if not len(x) == 2: continue
  id_, title = x
  title = title.lower()
  words = title.split()
  if words[0] in initial_words:
    goals.append((id_,title))

print 'Now we have %d goals'%len(goals)

# Some post processing
goals_ = []
for id_, goal in goals:
  goal = goal.rstrip('?')
  goal = re.sub('^matplotlib ', '', goal)
  goal = re.sub('^and ', '', goal)
  goal = re.sub('^/ ', '', goal)
  goal = re.sub('^- ', '', goal)
  goal = re.sub('^-- ', '', goal)
  goal = re.sub(' in matplotlib$', ' ', goal)
  goal = re.sub(' with matplotlib$', ' ', goal)
  goal = re.sub(' in python$', ' ', goal)
  goal = re.sub(' using matplotlib$', ' ', goal)
  goal = re.sub(' in python matplotlib$', ' ', goal)
  goals_.append((id_,goal))

goals = goals_

with open(relative_path('models/output/mpl_so_goals.txt'),'w') as writer:
  for id_, goal in sorted(goals):
    writer.write("%s\t%s\n"%(id_.encode('utf-8'), goal.encode('utf-8')))
