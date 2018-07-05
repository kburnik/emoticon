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
  if (this.o.fill) {
    ctx.fillStyle = this.o.fillStyle;
    ctx.fill();
    ctx.fillStyle = 'black';
  }
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
  ctx.fillStyle = 'white';
  ctx.fill();
  ctx.fillStyle = 'black';
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
    ctx.fillStyle = this.o.fillStyle;
    ctx.fill();
    ctx.fillStyle = 'black';
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
  var arc = this.o.arc;
  if (this.o.flip) {
    center = center.translate(0, center.y);
    arc = Arc.flip(arc);
  }
  ctx.arc(
      center.x,
      center.y,
      this.o.radius,
      Math.PI + arc.start,
      Math.PI + arc.end);
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
    lineWidth: scale * this.o.head.lineWidth,
    fill: this.o.head.fill,
    fillStyle: this.o.head.fillStyle
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
    flip: this.o.eyeBrows.flip,
    arc: this.o.eyeBrows.arc
  });

  var rightEyebrow = new Eyebrow({
    center: rightEye.o.center,
    offset: radius * this.o.eyeBrows.offset,
    radius: rightEye.o.borderRadius * this.o.eyeBrows.distance,
    lineWidth: scale * this.o.eyeBrows.lineWidth,
    flip: this.o.eyeBrows.flip,
    arc: this.o.eyeBrows.arc
  });

  var mouth = new Mouth({
    center: center,
    radius: radius * this.o.mouth.centerDistance,
    offset: radius * this.o.mouth.offset,
    lineWidth: scale * this.o.mouth.lineWidth,
    arc: this.o.mouth.arc,
    flip: this.o.mouth.flip,
    fill: this.o.mouth.fill,
    fillStyle: this.o.mouth.fillStyle,
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

Range.prototype.pick = function() {
  var items = (this.to - this.from) / this.step;
  for (var i = this.from; i <= this.to; i += this.step) {
    if (Math.random() < 1/items) {
      return i;
    }
  }
  return this.to;
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

Choice.prototype.pick = function() {
  return this.elements[Math.floor(Math.random() * this.elements.length)];
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

function draw(emoticon) {
  var canvas = document.getElementById("canvas");
  var ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "black";
  emoticon.render(ctx);
}

function saveCanvas(name) {
  canvas.toBlob(function(blob) {
    saveAs(blob, name);
  });
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

function drawGeneratedEmoticons(configSpace, prefix, max, doSave) {
  var pause = 100;
  var wait = 0;
  var total = 0;
  var max = max || 20;
  for (var i = 0; i < max; i++) {
    var config = select(configSpace);
    var emoticon = new Emoticon(config);
    (function(emoticon, name) {
      setTimeout(function() {
        draw(emoticon);
        if (doSave) {
          saveCanvas(name);
        }
      }, wait);
    })(emoticon, prefix + pad(total, 6) + ".png");
    wait += pause;
    total++;
  }
  console.log("Generating", total, "emoticons");
}

function select(config) {
  var out = Object.assign({}, config);
  var i = 0;
  while (true) {
    var changed = false;
    for (let key in out) {
      if (out[key] instanceof Range || out[key] instanceof Choice) {
        out[key] = out[key].pick()
        changed = true;
      } else if (typeof out[key] === 'object') {
        out[key] = select(out[key]);
      }
    }
    if (!changed || ++i > 10000) {
      break;
    }
  }
  return out;
}
