package main

import (
	"fmt"
	"os/exec"
)

type NVIDIA struct {
	model string
}

func (nvidia *NVIDIA) Run(prompt string) (string, error) {
	cmd := exec.Command("python3", "nvidia.py", prompt, nvidia.model)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	return string(output), nil
}

func main() {
	nvidia := NVIDIA{
		model: "meta/llama-3.2-3b-instruct",
	}
	output, err := nvidia.Run("Teach me gorilla websockets")
	if err != nil {
		panic(err)
	}
	fmt.Println(string(output))
}
