package main

import (
	"fmt"
	"encoding/csv"
	"os"
	"log"
	"math/rand"
	"strconv"
)

type Point struct {
	X, Y float64
}

type Vector struct {
	X, Y float64
}

type Particle struct {
	Point Point
	Velocity Vector
}

func (p *Particle) Tick(time float64) {
	p.Point = Point{
		X: p.Point.X + p.Velocity.X * time,
		Y: p.Point.Y + p.Velocity.Y * time,
	}
}

type Rectangle struct {
	origin, edge Point
}

func wouldCollide(a float64, b float64, lessThan bool) bool {
	if lessThan {
		return a < b
	}
	return a > b
}

func reflect(a float64, b float64) float64 {
	return 2 * b - a
}

func testCollision(point Point, particle *Particle, lessThan bool) {
	if wouldCollide(particle.Point.X, point.X, lessThan) {
		particle.Point.X = reflect(particle.Point.X, point.X)
		particle.Velocity.X *= -1
	}

	if wouldCollide(particle.Point.Y, point.Y, lessThan) {
		particle.Point.Y = reflect(particle.Point.Y, point.Y)
		particle.Velocity.Y *= -1
	}
}

func (r *Rectangle) collide(particle *Particle) {
	testCollision(r.origin, particle, true)
	testCollision(r.edge, particle, false)
}

func writeOutput(i int, p []Particle) {
	f, err := os.OpenFile("results.csv", os.O_RDWR|os.O_CREATE|os.O_APPEND, 0644)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	writer := csv.NewWriter(f)
	writer.Flush()

	data := make([][]string, len(p) + 1)

	for i := 0; i < len(p); i++ {
		data[i] = []string{strconv.Itoa(i), strconv.FormatFloat(p[i].Point.X, 'f', -1, 64), strconv.FormatFloat(p[i].Point.Y, 'f', -1, 64), strconv.FormatFloat(p[i].Velocity.X, 'f', -1, 64), strconv.FormatFloat(p[i].Velocity.Y, 'f', -1, 64)}
	}
	
	writer.WriteAll(data)
}

func main() {
	const BOX_SIZE = 100.0
	const PARTICLES = 100
	const ITERATIONS = 1000

	var rect = Rectangle{
		origin: Point{X: 0, Y: 0},
		edge: Point{X: BOX_SIZE, Y: BOX_SIZE},
	}

	var particles = make([]Particle, PARTICLES)

	for i := 0; i < PARTICLES; i++ {
		var float = float64(i)
		particles[i] = Particle{
			Point: Point{X: float / PARTICLES * BOX_SIZE, Y: rand.Float64() * BOX_SIZE},
			Velocity: Vector{X: rand.Float64() * 5, Y: rand.Float64() * 5},
		}
	}

	for i := 0; i < ITERATIONS; i++ {
		for j := range particles {
			particles[j].Tick(0.1)
			rect.collide(&particles[j])
		}
	
		writeOutput(i, particles)
	}
	fmt.Println("Done!")
}