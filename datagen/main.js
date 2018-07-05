



window.SAVE = true;
window.PROBABILITY = 0.7;
// drawGeneratedEmoticons(happy, "happy-");
// drawGeneratedEmoticons(surprised, "surprised-");
//drawGeneratedEmoticons(sad, "sad-");

const SCALE = 5;

const BOX = {
  width: 50,
  height: 50,
  margin: 1
}

const EYES_1 = {
  centerDistance: Choice.of([1/6, 1/8]),
  offset: Choice.of([1/4, 1/5, 1/7]),
  lineWidth: Choice.of([1/2, 1]),
  size: Choice.of([1.2, 1.4]),
  border: Choice.of([3, 4])
}

const EYES_2 = {
  centerDistance: Choice.of([1/6, 1/8]),
  offset: Choice.of([1/4, 1/5, 1/7]),
  lineWidth: Choice.of([1/2, 1]),
  size: Choice.of([1.5, 2, 3]),
  border: 1
}


const EYEBROWS_0 = {
  distance: 1.8,
  offset: 1/10,
  lineWidth: 0.0001,
  arc: Arc.of(1/4, 3/4),
};

const EYEBROWS_1 = {
  distance: 1.8,
  offset: Choice.of([1/10, 1/8]),
  lineWidth: Choice.of([1/2, 1]),
  arc: Arc.of(1/4, 3/4)
};

const EYEBROWS_2 = {
  distance: 1.35,
  offset: Choice.of([1/10, 1/8]),
  lineWidth: Choice.of([1/2, 1]),
  arc: Arc.of(1/5, 4/5)
};

// Surprise/fear.
const EYEBROWS_3 = {
  distance: Choice.of([1.6, 1.4]),
  offset: 0,
  lineWidth: Choice.of([1/2, 1]),
  arc: Arc.of(1/4, 3/4)
};


// Sad brows.
const EYEBROWS_4 = {
  distance: 1.3,
  offset: 1/10,
  lineWidth: Choice.of([1/2, 1]),
  arc: Arc.of(1/4, 3/4)
};

const HEAD_1 = {
  lineWidth: Choice.of([1/2, 1, 1.5]),
  fill: true,
  fillStyle: Choice.of(['white', 'yellow'])
}

const MOUTH_1 = {
  arc: Choice.of([ Arc.of(1/4, 3/4), Arc.of(1/5, 4/5), Arc.of(1/10, 9/10), ]),
  centerDistance: 3/5,
  offset: Choice.of([0, 1/10]),
  lineWidth: Choice.of([1/2, 1]),
  flip: false,
  fill: false
}

// Smiling (slightly surprised).
const MOUTH_2 = {
  arc: Choice.of([ Arc.of(1/4, 3/4), Arc.of(1/5, 4/5) ]),
  centerDistance: 3/5,
  offset: Choice.of([0, 1/10]),
  lineWidth: Choice.of([1/2, 1]),
  flip: false,
  fill: true,
  fillStyle: Choice.of(['white', 'black'])
}

// Frowning - sad.
const MOUTH_3 = {
  arc: Choice.of([ Arc.of(1/3, 2/3), Arc.of(1/4, 3/4), Arc.of(1/5, 4/5) ]),
  centerDistance: 3/5,
  offset: Choice.of([0, -1/8]),
  lineWidth: Choice.of([1/2, 1]),
  flip: true,
  fill: false
}

// Surprised mouth.
const MOUTH_4 = {
  arc: Arc.of(0, 2),
  centerDistance: Choice.of([0.1, 0.2]),
  offset: Choice.of([0.4, 0.6]),
  lineWidth: Choice.of([0.5, 1]),
  flip: false,
  fill: Choice.of([true, false]),
  fillStyle: 'black'
}


var happy = {
  scale: SCALE,
  box: BOX,
  head: Choice.of([HEAD_1]),
  eyes: Choice.of([EYES_1, EYES_2]),
  eyeBrows: Choice.of([EYEBROWS_0, EYEBROWS_1, EYEBROWS_2]),
  mouth: Choice.of([MOUTH_1])
}

var sad = {
  scale: SCALE,
  box: BOX,
  head: Choice.of([HEAD_1]),
  eyes: Choice.of([EYES_1, EYES_2]),
  eyeBrows: Choice.of([EYEBROWS_0, EYEBROWS_4]),
  mouth: Choice.of([MOUTH_3])
}

var surprised = {
  scale: SCALE,
  box: BOX,
  head: Choice.of([HEAD_1]),
  eyes: Choice.of([EYES_1, EYES_2]),
  eyeBrows: Choice.of([EYEBROWS_0, EYEBROWS_3]),
  mouth: Choice.of([MOUTH_4])
}

// draw(new Emoticon(select(surprised)));

// drawGeneratedEmoticons(surprised, "surprised-", 400, true);
// drawGeneratedEmoticons(sad, "sad-", 400, true);
// drawGeneratedEmoticons(happy, "happy-", 400, true);
