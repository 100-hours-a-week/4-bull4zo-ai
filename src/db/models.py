from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Word(Base):
    __tablename__ = 'words'
    id = Column(Integer, primary_key=True)
    word = Column(String(100))
    operator_id = Column(String(100))

class WordInfo(Base):
    __tablename__ = 'word_infos'
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'))
    info = Column(Text)

class Vote(Base):
    __tablename__ = 'votes'
    id = Column(Integer, primary_key=True)
    word_id = Column(Integer, ForeignKey('words.id'))
    content = Column(Text)
    result = Column(String(20))
