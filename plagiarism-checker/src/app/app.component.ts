import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { PlagiarismService } from './plagiarism.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, CommonModule, FormsModule, HttpClientModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'plagiarism-checker';
  code1 = '';
  code2 = '';
  result: number | null = null;
  similarity: number = 0;
  error: string | null = null;

  constructor(private plagiarismService: PlagiarismService) {}

  checkPlagiarism() {
    this.error = null;
    this.result = null;
    if (!this.code1.trim() || !this.code2.trim()) {
      this.error = 'Both code snippets are required.';
      return;
    }

    this.plagiarismService.check(this.code1, this.code2).subscribe({
      next: res => {
        this.result = res.prediction;
        this.similarity = res.similarity;
      },
      error: err => {
        this.error = err.error?.message || 'An error occurred while checking similarity.';
      }
    });
  }
}
