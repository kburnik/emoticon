function Point(x, y) {
  this.x = x;
  this.y = y;
}

function Arc(start, end) {
  this.start = start;
  this.end = end;
}

Arc.flip = function(arc) {
  return new Arc(arc.start + Math.PI, arc.end + Math.PI);
}

Arc.of = function (a, b) {
  return new Arc(Math.PI * a, Math.PI * b);
}

Point.prototype.translate = function(dx, dy) {
  return new Point(this.x + dx, this.y + dy);
}

function Head(options) {
  this.o = options;
}

Head.prototype.render = function(ctx) {
  ctx.beginPath();
  ctx.lineWidth = this.o.lineWidth;
  ctx.arc(
      this.o.center.x,
      this.o.center.y,
      this.o.radius,
      0,
      2 * Math.PI);
  ctx.stroke();
}

function Eye(options) {
  this.o = options;
}

Eye.prototype.render = function(ctx) {
  var center = this.o.center.translate(0, this.o.offset);
  // Border of the eye.
  ctx.beginPath();
  ctx.lineWidth = this.o.lineWidth;
  ctx.arc(
      center.x,
      center.y,
      this.o.borderRadius,
      0,
      2 * Math.PI,
      false);
  ctx.stroke();

  // Pupil.
  ctx.beginPath();
  ctx.arc(
      center.x,
      center.y,
      this.o.radius,
      0,
      2 * Math.PI,
      false);
  ctx.fill();
  ctx.stroke();
}

function Mouth(options) {
  this.o = options;
}

Mouth.prototype.render = function(ctx) {
  ctx.beginPath();
  ctx.lineWidth = this.o.lineWidth;
  var center = this.o.center.translate(0, this.o.offset);
  var arc = this.o.arc;
  if (this.o.flip) {
    center = center.translate(0, center.y);
    arc = Arc.flip(arc);
  }
  ctx.arc(
      center.x,
      center.y,
      this.o.radius,
      arc.start,
      arc.end);
  if (this.o.fill) {
    ctx.fill();
  }
  ctx.stroke();
}


function Eyebrow(options) {
  this.o = options;
}

Eyebrow.prototype.render = function(ctx) {
  var center = this.o.center.translate(0, this.o.offset)
  ctx.beginPath();
  ctx.lineWidth = this.o.lineWidth;
  ctx.arc(
      center.x,
      center.y,
      this.o.radius,
      Math.PI + this.o.arc.start,
      Math.PI + this.o.arc.end);
  ctx.stroke();
}

function Emoticon(options) {
  this.o = options;
}

Emoticon.prototype.render = function(ctx) {
  var scale = this.o.scale;
  var width = scale * this.o.box.width;
  var height = scale * this.o.box.height;
  var margin = scale * this.o.box.margin;

  var center = new Point(width / 2, height / 2);
  var radius = width / 2 - margin * 2;

  var head = new Head({
    center: center,
    radius: radius,
    lineWidth: scale * this.o.head.lineWidth
  });

  var leftEye = new Eye({
    center: center.translate(
        -width * this.o.eyes.centerDistance,
        -height * this.o.eyes.centerDistance),
    offset: radius * this.o.eyes.offset,
    radius: scale * this.o.eyes.size,
    lineWidth: scale * this.o.eyes.lineWidth,
    borderRadius: scale * this.o.eyes.size * this.o.eyes.border
  });

  var rightEye = new Eye({
    center: center.translate(
        width * this.o.eyes.centerDistance,
        -height * this.o.eyes.centerDistance),
    offset: radius * this.o.eyes.offset,
    radius: scale * this.o.eyes.size,
    lineWidth: scale * this.o.eyes.lineWidth,
    borderRadius: scale * this.o.eyes.size * this.o.eyes.border
  });

  var leftEyebrow = new Eyebrow({
    center: leftEye.o.center,
    offset: radius * this.o.eyeBrows.offset,
    radius: leftEye.o.borderRadius * this.o.eyeBrows.distance,
    lineWidth: scale * this.o.eyeBrows.lineWidth,
    arc: this.o.eyeBrows.arc
  });

  var rightEyebrow = new Eyebrow({
    center: rightEye.o.center,
    offset: radius * this.o.eyeBrows.offset,
    radius: rightEye.o.borderRadius * this.o.eyeBrows.distance,
    lineWidth: scale * this.o.eyeBrows.lineWidth,
    arc: this.o.eyeBrows.arc
  });

  var mouth = new Mouth({
    center: center,
    radius: radius * this.o.mouth.centerDistance,
    offset: radius * this.o.mouth.offset,
    lineWidth: scale * this.o.mouth.lineWidth,
    arc: this.o.mouth.arc,
    flip: this.o.mouth.flip,
    fill: this.o.mouth.fill
  });

  var faceParts = [
    head, leftEye, rightEye, leftEyebrow, rightEyebrow, mouth
  ];

  faceParts.forEach(p => p.render(ctx));
}

function ConfigSpace(options) {
  this.o = options;
}


function Range(from, to, step) {
  this.from = from;
  this.to = to;
  this.step = step;
}

Range.prototype.generate = function*() {
  for (var i = this.from; i <= this.to; i += this.step) {
    yield i;
  }
}

Range.of = function(from, to, step) {
  return new Range(from, to, step);
}

function Choice(elements) {
  this.elements = elements;
}

Choice.of = function(elements) {
  return new Choice(elements);
}

Choice.prototype.generate = function * () {
  yield * this.elements;
}

ConfigSpace.prototype.generate = function*(probability) {
  function * generate(options, skipKeys, depth) {
    var key = null;
    for (var candidateKey in options) {
      if (candidateKey in skipKeys) {
        continue;
      }
      key = candidateKey;
      break;
    }

    if (key == null) {
      if (Math.random() < probability) {
        yield JSON.parse(JSON.stringify(options));
      }
      return;
    }

    var value = options[key];
    var elements = [value];

    if (value instanceof Range || value instanceof Choice) {
      elements = value.generate();
    } else if (typeof value === 'object') {
      elements = generate(options[key], {}, depth + 1);
    } else {
      // Keep as is.
    }

    var oldValue = options[key];
    skipKeys[key] = true;

    for (let subValue of elements) {
      options[key] = subValue;
      yield * generate(options, skipKeys, depth);
    }

    options[key] = oldValue;
    delete skipKeys[key];
  }

  yield * generate(this.o, {}, 0);
};

var sampleEmoticon = new Emoticon({
  scale: 10,
  box: {
    margin: 1,
    width: 50,
    height: 50,
  },
  head: {
    lineWidth: 1,
  },
  eyes: {
    centerDistance: 1/6,
    offset: 1/5,
    lineWidth: 1/2,
    left: {
      size: 1.5,
      border: 4,
    },
    right: {
      size: 1.5,
      border: 4
    }
  },
  eyeBrows: {
    distance: 1.5,
    offset: 1/7,
    lineWidth: 1,
    left: {
      arc: Arc.of(1/4, 3/4)
    },
    right: {
      arc: Arc.of(1/4, 3/4)
    }
  },
  mouth: {
    arc: Arc.of(0, 2),
    centerDistance: 1/5,
    offset: 1/2,
    lineWidth: 1/2,
    flip: false,
    fill: true
  }
});

var happyBase = {
  head: {
    lineWidth: 1,
  },
  eyes: {
    centerDistance: 1/8,
    offset: 1/5,
    lineWidth: 1/2,
    size: 1.2,
    border: 4
  },
  eyeBrows: {
    distance: 1.8,
    offset: 1/10,
    lineWidth: 1/2,
    arc: Arc.of(1/4, 3/4),
  },
  mouth: {
    arc: Arc.of(1/4, 3/4),
    centerDistance: 3/5,
    offset: 0,
    lineWidth: 1/2,
    flip: false,
    fill: false
  }
};

var happy = new ConfigSpace(Object.assign({}, happyBase, {

  head: {
    lineWidth: Choice.of([0.1, 0.5, 1]),
  },

  eyes: {
    centerDistance: Choice.of([1/8, 1/7, 1/5]),
    offset: Choice.of([1/10, 1/6]),
    lineWidth: 1/2,
    size: Choice.of([0.8, 1.5, 3]),
    border: Choice.of([0.1, 1, 2])
  },

  eyeBrows: {
    distance: Choice.of([1.8, 1.9]),
    offset: 1/10,
    lineWidth: Choice.of([1/1000, 1/2, 1]),
    arc: Choice.of([ Arc.of(1/4, 3/4), Arc.of(1/3, 2/3) ])
  },

  mouth: {
    arc: Choice.of([ Arc.of(1/4, 3/4), Arc.of(1/3, 2/3) ]),
    centerDistance: Choice.of([3/5, 4/5]),
    offset: 0,
    lineWidth: Choice.of([1/2, 1]),
    flip: false,
    fill: Choice.of([true, false])
  }

}));


var surprised = new ConfigSpace(Object.assign({}, happyBase, {

  head: {
    lineWidth: Choice.of([0.1, 0.5, 1]),
  },

  eyes: {
    centerDistance: Choice.of([1/8, 1/7, 1/5]),
    offset: Choice.of([1/10, 1/6]),
    lineWidth: 1/2,
    size: Choice.of([0.8, 1.5, 3]),
    border: Choice.of([0.1, 1, 2])
  },

  eyeBrows: {
    distance: 1.2,
    offset: 1/50,
    lineWidth: Choice.of([1/1000, 1/2, 1]),
    arc: Choice.of([ Arc.of(1/4, 3/4) ])
  },

  mouth: {
    arc: Arc.of(0, 2),
    centerDistance: Choice.of([0.1, 0.2]),
    offset: Choice.of([0.4, 0.6]),
    lineWidth: Choice.of([0.5, 1]),
    flip: false,
    fill: Choice.of([true, false])
  }
}));




var sadBase = {
  head: {
    lineWidth: 1,
  },
  eyes: {
    centerDistance: 1/8,
    offset: 1/5,
    lineWidth: 1/2,
    size: 1.2,
    border: 1
  },
  eyeBrows: {
    distance: 1.8,
    offset: 1/10,
    lineWidth: 1/2,
    arc: Arc.of(1/4, 3/4),
  },
  mouth: {
    arc: Arc.of(1/4, 3/4),
    centerDistance: 3/5,
    offset: 0,
    lineWidth: 1/2,
    flip: false,
    fill: false
  }
};

var sad = new ConfigSpace(Object.assign({}, sadBase, {

  head: {
    lineWidth: Choice.of([0.1, 0.5, 1]),
  },

  eyes: {
    centerDistance: Choice.of([1/8, 1/7, 1/5]),
    offset: Choice.of([1/10, 1/6]),
    lineWidth: 1/2,
    size: Choice.of([0.8, 1.5, 3]),
    border: Choice.of([0.1, 1, 2])
  },

  eyeBrows: {
    distance: Choice.of([1.8, 1.9]),
    offset: 1/10,
    lineWidth: Choice.of([1/1000, 1/2, 1]),
    arc: Choice.of([ Arc.of(1/4, 3/4), Arc.of(1/3, 2/3) ])
  },

  mouth: {
    arc: Choice.of([ Arc.of(1/4, 3/4) ]),
    centerDistance: Choice.of([3/5, 2/5]),
    offset: 0,
    lineWidth: Choice.of([1/2, 1]),
    flip: true,
    fill: false
  }

}));


function draw(emoticon, name) {
  var canvas = document.getElementById("canvas");
  var ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "black";
  emoticon.render(ctx);

  if (window.SAVE) {
    console.log("Saving", name);
    canvas.toBlob(function(blob) {
      saveAs(blob, name);
    });
  }

}

function * generateEmoticons(configSpace) {
  var defaults = {
    scale: 5,
    box: {
      margin: 1,
      width: 50,
      height: 50,
    }
  };
  for (let subConfig of configSpace.generate(window.PROBABILITY)) {
    var config = Object.assign({}, defaults, subConfig);
    yield new Emoticon(config);
  }
}

function pad(num, size) {
  var s = "000000000" + num;
  return s.substr(s.length-size);
}

function drawGeneratedEmoticons(configSpace, prefix) {
  var pause = 100;
  var wait = 0;
  var total = 0;
  for (let emoticon of generateEmoticons(configSpace)) {
    (function(emoticon, name) {
      setTimeout(function() {
        draw(emoticon, name);
      }, wait);
    })(emoticon, prefix + pad(total, 6) + ".png");
    wait += pause;
    total++;
  }
  console.log("Generating", total, "emoticons");
}


// drawGeneratedEmoticons(happy, "happy-");
// drawGeneratedEmoticons(surprised, "surprised-");
window.SAVE = true;
window.PROBABILITY = 0.7;
drawGeneratedEmoticons(sad, "sad-");
