from config import db


class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(120), nullable=False)
    languages = db.Column(db.String(100), nullable=True)

    def __repr__(self):
        return '<Job %r>' % self.title

    def __str__(self):
        return self.title


class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(100), nullable=False)
    answer = db.Column(db.String(100), nullable=False)
    options = db.Column(db.String(100), nullable=True)

    def __repr__(self):
        return '<Question %r>' % self.text

    def __str__(self):
        return self.text


db.create_all()
db.session.commit()
