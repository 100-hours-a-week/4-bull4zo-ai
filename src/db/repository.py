from src.db.models import Word, WordInfo, Vote
from sqlalchemy.orm import Session

class WordRepository:
    def save(self, session: Session, word: Word):
        session.add(word)
        session.commit()
        session.refresh(word)
        return word

    def get(self, session: Session, word_id: int):
        return session.query(Word).filter_by(id=word_id).first()

    def delete(self, session: Session, word_id: int):
        word = self.get(session, word_id)
        if word:
            session.delete(word)
            session.commit()
        return word

class WordInfoRepository:
    def save(self, session: Session, word_info: WordInfo):
        session.add(word_info)
        session.commit()
        session.refresh(word_info)
        return word_info

    def get(self, session: Session, word_info_id: int):
        return session.query(WordInfo).filter_by(id=word_info_id).first()

    def get_by_word_id(self, session: Session, word_id: int):
        return session.query(WordInfo).filter_by(word_id=word_id).all()

    def delete_by_word_id(self, session: Session, word_id: int):
        infos = self.get_by_word_id(session, word_id)
        for info in infos:
            session.delete(info)
        session.commit()
        return infos

class VoteRepository:
    def save(self, session: Session, vote: Vote):
        session.add(vote)
        session.commit()
        session.refresh(vote)
        return vote

    def get(self, session: Session, vote_id: int):
        return session.query(Vote).filter_by(id=vote_id).first()

    def get_by_word_id(self, session: Session, word_id: int):
        return session.query(Vote).filter_by(word_id=word_id).all()

    def update_result(self, session: Session, vote_id: int, result: str):
        vote = self.get(session, vote_id)
        if vote:
            vote.result = result
            session.commit()
            session.refresh(vote)
        return vote

    def delete_by_word_id(self, session: Session, word_id: int):
        votes = self.get_by_word_id(session, word_id)
        for vote in votes:
            session.delete(vote)
        session.commit()
        return votes 